import os
import json
import logging
import psycopg2
import textwrap

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    room_io,
    UserInputTranscribedEvent,
    ConversationItemAddedEvent,
    ChatContext,
    ChatMessage,
    StopResponse,
    function_tool,
    RunContext,
)
from qdrant_client import QdrantClient
from openai import OpenAI
from langdetect import detect
from livekit.plugins import cartesia, openai, silero, elevenlabs
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env")
ELEVENLABS_API = os.getenv("ELEVENLABS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def open_connection():
    conn = psycopg2.connect(
        database="rag",
        user="postgres",
        host="host.docker.internal",
        password="postgres",
        port=5435,
    )
    return conn


def get_embeddings(text):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(input=text, model=os.getenv("EMBEDDING_MODEL"))

    return response.data[0].embedding


def get_qdrant_client():
    client = QdrantClient(url=os.getenv("QDRANT_URL"))

    return client


# --------------------------------------------------
# VAD tuning profiles
# --------------------------------------------------
# activation_threshold: higher -> more conservative, less background noise
#                       reaches the STT (but may miss very soft speech).
# min_speech_duration:  higher -> rejects short transient noise (clicks,
#                       coughs, distant chatter) before it opens a speech chunk.
#
# ACTIVE  -> conservative. Used on startup and in "vision active mode" so only
#            the user talking to the agent reaches the STT, not the room noise.
# PASSIVE -> sensitive. Used in "vision passive mode" meeting capture so all
#            speakers (including softer/farther ones) are transcribed.
VAD_ACTIVE_PROFILE = dict(activation_threshold=0.7, min_speech_duration=0.15)
VAD_PASSIVE_PROFILE = dict(activation_threshold=0.5, min_speech_duration=0.05)


# --------------------------------------------------
# State machine
# --------------------------------------------------
# mode:             "active" | "passive"
# record_status:    "idle"   | "recording" | "ended"
# meeting_status:   "idle"   | "ongoing"   | "ended"
# --------------------------------------------------


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            llm=openai.LLM(model=os.getenv("OPENAI_MODEL")),
            instructions=textwrap.dedent(
                """\
                You are a polite assistant that helps company employees manage
                meetings and store meeting recordings.

                When you first start, greet the user and ask them to choose a mode
                by saying "vision active mode" or "vision passive mode".

                In PASSIVE mode you are silent. You do NOT speak or respond.
                The meeting transcript is being captured in the background.

                In ACTIVE mode you interact normally. If a recording exists you
                should ask whether the user wants to store it. Use the
                'insert_meeting_recording' tool when they confirm.

                If you are ever asked about SureFlow, use 'sureflow_information_retrieval' tool.
                """
            ),
        )
        # state
        self.mode: str = "active"
        self.record_status: str = "idle"
        self.meeting_status: str = "idle"
        self.meeting_title: str = ""
        self.transcript_lines: list[str] = []

    # -- adjust how much sound reaches the STT for the current mode ----------------------
    def _apply_vad_profile(self, profile: dict) -> None:
        vad = self.session.vad
        if vad is None:
            return
        vad.update_options(**profile)
        logger.info(f"Applied VAD profile: {profile}")

    # -- hook: runs before the LLM replies -----------------------------------------------
    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ):
        text = (new_message.text_content or "").lower()

        # --- Switch to PASSIVE MODE ------------------------------------
        if "vision" in text and "passive mode" in text:
            self.mode = "passive"
            self.record_status = "recording"
            self.meeting_status = "ongoing"

            # Meeting capture: widen VAD so all speakers are transcribed.
            self._apply_vad_profile(VAD_PASSIVE_PROFILE)

            # Ask for a meeting title before going quiet
            await self.session.say(
                "Passive mode activated. What would you like the meeting title to be?"
                "You can let me know the title now or what the meeting would be about - I'll capture it for you and go silent."
            )
            raise StopResponse()

        # --- Switch to ACTIVE MODE -------------------------------------
        if "vision" in text and "active mode" in text:
            self.mode = "active"

            # Conversing 1:1: tighten VAD so background noise stays out of the STT.
            self._apply_vad_profile(VAD_ACTIVE_PROFILE)

            if self.record_status == "recording":
                # Inject context so the LLM knows to ask about storing
                turn_ctx.add_message(
                    role="assistant",
                    content=(
                        "[System] The user switched to active mode."
                        "A meeting is in progress."
                        "How can I help you?"
                    ),
                )
                return

            if self.record_status == "ended":
                turn_ctx.add_message(
                    role="assistant",
                    content=(
                        "[System] The recording has already been stored."
                        "Ask: 'What else can I help you with?'"
                    ),
                )
                return
            return

        if self.mode == "passive":
            # Capture meeting title if we do not have one yet
            if not self.meeting_title:
                self.meeting_title = new_message.text_content or "Untitled Meeting"
                await self.session.say(
                    f"Got it - '{self.meeting_title}'. Going silent now."
                    "Say 'vision active mode' when the meeting is over."
                )
                raise StopResponse

            # Otherwise just accumlate the transcript - no LLM reply
            self.transcript_lines.append(new_message.text_content or "")
            turn_ctx.items.append(new_message)

            await self.update_chat_ctx(turn_ctx)
            raise StopResponse()

    @function_tool()
    async def insert_meeting_recording(
        self,
        context: RunContext,
        meeting_title: str,
        meeting_content: str,
    ) -> str:
        """Insert a meeting trasncript with its embedding into pgvector for RAG retrieval.

        Args:
            meeting_title: The title or name of the meeting, e.g. 'Q3 Revenue Review'.
            meeting_content: The transcript or text content from the meeting to store.
        """
        import asyncio

        def _do_insert():
            conn = open_connection()
            cursor = conn.cursor()
            meeting_title_emb = get_embeddings(meeting_title)
            embeddings = get_embeddings(meeting_content)
            cursor.execute(
                """
                INSERT INTO meeting_recording(
                    context, meeting_title, meeting_title_emb, embedding
                ) VALUES (%s, %s, %s, %s)
                """,
                (meeting_content, meeting_title, meeting_title_emb, embeddings),
            )
            conn.commit()
            cursor.close()
            conn.close()

        await asyncio.get_event_loop().run_in_executor(None, _do_insert)
        return f"Stored meeting '{meeting_title}' in the database"

    # Sureflow RAG
    @function_tool()
    async def sureflow_information_retrieval(
        self,
        context: RunContext,
        query: str,
    ) -> str:
        """You can use this tool if you are asked about Sureflow and if you do not know anything about SureFlow.

        Args:
            query: The doubt you have about SureFlow.
        """
        import asyncio

        def _do_retrieve():
            client = get_qdrant_client()
            embedded_query = get_embeddings(query)
            search_result = client.query_points(
                collection_name="sureflow",
                query=embedded_query,
                with_payload=True,
                limit=3,
            ).points

            chunks = []
            for point in search_result:
                chunks.append(
                    f"Section: {point.payload['section']}"
                    f"\nContent: {point.payload['content']}"
                )
            return "\n\n---\n\n".join(chunks)

        result = await asyncio.get_event_loop().run_in_executor(None, _do_retrieve)
        return result


server = AgentServer()


def prewarm(proc: JobProcess):
    # Start conservative: the agent boots in "active" mode, so reject room
    # noise from the very first turn. Passive mode loosens this at runtime.
    proc.userdata["vad"] = silero.VAD.load(**VAD_ACTIVE_PROFILE)


server.setup_fnc = prewarm


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Per-language Cartesia voices. Swap these IDs for the voices you prefer.
    voice_by_lang = {
        "en": "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",  # English voice
        "fr": "65b25c5d-ff07-4687-a04c-da2f43ef6fa9",  # French voice
    }
    default_lang = "en"

    session = AgentSession(
        stt=openai.STT(
            model="gpt-4o-mini-transcribe",
        ),
        tts=cartesia.TTS(
            model="sonic-3",
            language=default_lang,
            voice=voice_by_lang[default_lang],
            speed=1.0
        ),
        # tts=elevenlabs.TTS(model="eleven_flash_v2_5", voice_id="Xb7hH8MSUJpSbSDYk0k2"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Track the active language so we only update the TTS when it actually changes.
    current_lang = default_lang

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(event: UserInputTranscribedEvent):
        logger.info(
            f"User input transcribed: {event.transcript},"
            f"language: {event.language},"
            f"speaker id: {event.speaker_id}"
        )

        # Normalize e.g. "fr-FR" -> "fr", then switch voice if the language changed.
        nonlocal current_lang

        try:
            detected = detect(event.transcript)
        except Exception:
            detected = "en"

        
        lang = detected.split("-")[0]
        if lang in voice_by_lang and lang != current_lang:
            current_lang = lang
            session.tts.update_options(
                language=current_lang,
                voice=voice_by_lang[current_lang],
            )
            logger.info(f"Switched Cartesia voice to '{lang}'")

    @session.on("conversation_item_added")
    def on_conversation_item_added(event: ConversationItemAddedEvent):
        if not isinstance(event.item, ChatMessage):
            return
        logger.info(
            f"Conversation item added from {event.item.role}. interrupted: {event.item.interrupted}"
        )

        for content in event.item.content:
            if isinstance(content, str):
                logger.info(f"   - text: {content}")

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            text_output=room_io.TextOutputOptions(
                json_format=True,
                sync_transcription=True,
            )
        ),
    )

    await session.generate_reply(
        
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)

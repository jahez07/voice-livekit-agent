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
from livekit.plugins import cartesia, openai, silero, elevenlabs
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env")
ELEVENLABS_API = os.getenv('ELEVENLABS_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def open_connection():
    conn = psycopg2.connect(
        database = 'rag',
        user = 'postgres',
        host = 'host.docker.internal',
        password = 'postgres',
        port = 5435
    )
    return conn

def get_embeddings(text):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(input=text,model='text-embedding-3-small')

    return response.data[0].embedding

def get_qdrant_client():
    client = QdrantClient(url="http://qdrant:6333")

    return client


def get_sureflow_info(query, client):
    search_result = client.query_points(
        collection_name="sureflow",
        query=query,
        with_payload=True,
        limit=3
    ).points

    result = []
    for point in search_result:
        result.append(
            f"Section: {point.payload['section']}\nContent: {point.payload['content']}"
        )

    return result
    

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
            llm=openai.LLM(model="gpt-4o-mini"),
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


    # -- hook: runs before the LLM replies -----------------------------------------------
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage):
        text = (new_message.text_content or "").lower()

        # --- Switch to PASSIVE MODE ------------------------------------
        if "vision" in text and "passive mode" in text:
            self.mode = "passive"
            self.record_status = "recording"
            self.meeting_status = "ongoing"

            # Ask for a meeting title before going quiet
            await self.session.say(
                "Passive mode activated. What would you like the meeting title to be?"
                "You can let me know the title now or what the meeting would be about - I'll capture it for you and go silent."
            )
            raise StopResponse()
        
        # --- Switch to ACTIVE MODE -------------------------------------
        if "vision" in text and "active mode" in text:
            self.mode = "active"

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
                """, (meeting_content, meeting_title, meeting_title_emb, embeddings)
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
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt=openai.STT(
            model="gpt-4o-mini-transcribe",
        ),
        tts=cartesia.TTS(model="sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        #tts=elevenlabs.TTS(model="eleven_flash_v2_5", voice_id="Xb7hH8MSUJpSbSDYk0k2"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(event: UserInputTranscribedEvent):
        print(f"User input transcribed: {event.transcript},"
              f"language: {event.language},"
              f"speaker id: {event.speaker_id}")
        
    @session.on("conversation_item_added")
    def on_conversation_item_added(event: ConversationItemAddedEvent):
        if not isinstance(event.item,ChatMessage):
            return
        print(f"Conversation item added from {event.item.role}. interrupted: {event.item.interrupted}")

        for content in event.item.content:
            if isinstance(content, str):
                print(f"   - text: {content}")

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

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)

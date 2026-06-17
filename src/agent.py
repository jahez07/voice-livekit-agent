import os
import re
import logging
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
import psycopg2
import os

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
        host = 'localhost',
        password = 'postgres',
        port = 5435
    )
    return conn

def get_embeddings(text):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(
        input=text,
        model='text-embedding-3-small'
    )
    embedding = response.data[0].embedding

    return embedding



class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            llm=openai.LLM(model="gpt-4o-mini"),
            instructions=textwrap.dedent(
                """\
                You are polite assistant that helps the company employees to manage meetings and record meeting recordings to store it.
                When you spun up for the first time, you must always ask the user if they want "active mode" or "passive mode" and for them to select, they have to say either "vision active mode" or "vision passive mode"

                Whichever mode you are in, until the meeting is over, you have to store the meeting embedding to a pgvector db. You can use the 'insert_meeting_recording' tool to insert meeting recording embedding.
                """
            ),
        )
        self.active = True
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage):
        text = (new_message.text_content or "").lower()

        if "vision" in text and "active mode" in text:
            self.active = True
            return
        
        if ("vision" in text and "passive mode") or ("passive mode") in text:
            self.active = False
            await self.session.say("Going quiet. Say 'vision active mode' when you need me.")
            raise StopResponse()
        
        if not self.active:
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
            embeddings = get_embeddings(context)
            cursor.execute(
                """
                INSERT INTP meeting_recording(
                    context, meeting_title, meeting_title_emb, embedding
                ) VALUES (%s, %s, %s, %s)
                """, (meeting_content, meeting_title, meeting_title_emb, embeddings)
            )
            conn.commit()
            cursor.close()
            conn.close()
        
        await asyncio.get_event_loop().run_in_executor(None, _do_insert)
        return f"Stored meeting '{meeting_title}' in the database"

    


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
        tools=[insert_meeting_recording]
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

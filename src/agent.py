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
    inference
)

from livekit.plugins import cartesia, openai, silero, elevenlabs
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env")

ELEVENLABS_API = os.getenv('ELEVENLABS_API_KEY')

def _matches(self, text: str, mode: str) -> bool:
    t = re.sub(r"[^a-z\s]", "", text.lower())
    return ("vision" in t or "division" in t) and mode in t and "mode" in t


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            llm=openai.LLM(model="gpt-4o-mini"),
            instructions=textwrap.dedent(
                """\
                Ask the user to select between
                active mode
                passive mode
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
        #tts=cartesia.TTS(model="sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        tts=elevenlabs.TTS(model="eleven_flash_v2_5", voice_id="Xb7hH8MSUJpSbSDYk0k2"),
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

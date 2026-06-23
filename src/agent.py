import os
import json
import yaml
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
    inference
)
from dataclasses import dataclass, field
from qdrant_client import QdrantClient
from typing import Annotated
from pydantic import Field
from openai import OpenAI
from langdetect import detect
from livekit.plugins import cartesia, openai, silero, elevenlabs
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env")
ELEVENLABS_API = os.getenv("ELEVENLABS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

voices = {
    "greeter": "e07c00bc-4134-4eae-9ea4-1a55fb45746b",
    "reservation": "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
    "takeaway": "5ee9feff-1265-424a-9d7f-8e4d431a12c7",
    "checkout": "a167e0f3-df7e-4d52-a9c3-f949145efdab",
}

# Class to collect data
@dataclass
class UserData:
    customer_name: str | None = None
    customer_phone: str | None = None

    reservation_time: str | None = None

    order: list[str] | None = None

    customer_credit_card: str | None = None
    customer_credit_card_expiry: str | None = None
    customer_credit_card_cvv: str | None = None

    expense: float | None = None
    checked_out: bool | None = None

    agents: dict[str, Agent] = field(default_factory=dict)
    prev_agent: Agent | None = None

    def summarize(self) -> str:
        data = {
            "customer_name": self.customer_name or "unknown",
            "customer_phone": self.customer_phone or "unkown",
            "reservation_time": self.reservation_time or "unknown",
            "order": self.order or "unknown",
            "credit_card": {
                "number": self.customer_credit_card or "unknown",
                "expiry": self.customer_credit_card_expiry or "unknown",
                "cvv": self.customer_credit_card_expiry or "unknown",
            }
            if self.customer_credit_card
            else None,
            "expense": self.expense or "unknown",
            "checked_out": self.checked_out or False,
        }
        return yaml.dump(data)
    
RunContext_T = RunContext[UserData]


# Shared tools
@function_tool()
async def update_name(
    name: Annotated[str, Field(description="The customer's name")],
    context: RunContext_T,
) -> str:
    """Called when the user provides their name.
    Confirm the spelling with the user before calling this function."""
    userdata = context.userdata
    userdata.customer_name = name
    return f"The name is updated to {name}"

@function_tool()
async def update_phone(
    phone: Annotated[str, Field(description="The customer's phone number")],
    context: RunContext_T,
) -> str:
    """Called when the user provides theor phone number.
    Confirm the phone number with the user before calling this function"""
    userdata = context.userdata
    userdata.customer_phone = phone
    return f"The phone number is updated to {phone}"

@function_tool()
async def to_greeter(context: RunContext_T) -> Agent:
    """Called when user asks unrelated questions or requests
    any other services not in your job description."""
    curr_agent: BaseAgent = context.session.current_agent
    return await curr_agent._transfer_to_agent("greeter", context)


# Base agent
class BaseAgent(Agent):
    """Base class that every specialist agent extends. Handles two things
    that all agents need: copying context from the previous agent on entry,
    and transferring control to the next agent on exit."""

    async def on_enter(self) -> None:
        """Called by the framework when this agent becomes active."""
        agent_name = self.__class__.__name__
        logger.info(f"entering task {agent_name}")

        userdata: UserData = self.session.userdata
        chat_ctx = self.chat_ctx.copy()

        # Copy the last few turns from the previous agent so this agent
        # has conversational continuity without carrying the full history.=
        # truncate(max_item=6)keeps context growth bounded across handoffs.
        if isinstance(userdata.prev_agent, Agent):
            truncated_chat_ctx = userdata.prev_agent.chat_ctx.copy(
                exclude_instructions=True,
                exclude_function_call=False,
                exclude_handoff=True,
                exclude_config_update=True,
            ).truncate(max_items=6)
            existing_ids = {item.id for item in chat_ctx.items}
            items_copy = [item for item in truncated_chat_ctx.items if item.id not in existing_ids]
            chat_ctx.items.extend(items_copy)
        
        # Inject the serialized UserData as a system message so this agent
        # knows the customer name, order, and other collected data.
        chat_ctx.add_message(
            role="system",
            content=f"You are {agent_name} agent. Current user data is {userdata.summarize()}",
        )
        await self.update_chat_ctx(chat_ctx)
        self.session.generate_reply(tool_choice="none")

    async def _transfer_to_agent(self, name: str, context: RunContext_T) -> tuple[Agent, str]:
        """Look up the next agent by name from the shared registry and hand
        off control. Returning an (Agent, str) tuple from a tool triggers
        the framework's handoff mechanism."""
        userdata = context.userdata
        current_agent = context.session.current_agent
        next_agent = userdata.agents[name]
        userdata.prev_agent = current_agent

        return next_agent, f"Transferring to {name}."


# Greeting Agent
class Greeter(BaseAgent):
    def __init__(self, menu: str) -> None:
        super().__init__(
            instructions=(
                f"You are a friendly restaurant receptionist. The menu is: {menu}\n"
                "Your jobs are to greet the caller and understand if they want to "
                "make a reservation or order takeaway. Guide them to the right agent using tools."
            ),
            llm=openai.LLM(model=os.getenv("OPENAI_MODEL"), parallel_tool_calls=False),
            tts=cartesia.TTS(model="sonic-3", voice=voices["greeter"]),
        )
        self.menu = menu
    
    @function_tool()
    async def to_reservation(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when user wants to make or update a reservation.
        This function handles a transitioning to the reservation agent
        who will collect the necessary details like reservation time,
        customer name and phone number."""
        return await self._transfer_to_agent("reservation", context)
    
    @function_tool()
    async def to_takeaway(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the user wants to place a takeaway order.
        This includes handling orders for pickup, delivery, or when the user wants to
        proceed to checkout with their existing order."""
        return await self._transfer_to_agent("takeaway", context)
    

# Reservation Agent
class Reservation(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a reservation agent at a restaurant. Your jobs are to ask for "
            "the reservation time, then customer's name, and phone number. Then "
            "confirm the reservation details with the customer.",
            tools=[update_name, update_phone, to_greeter],
            tts=cartesia.TTS(model="sonic-3", voice=voices["reservation"]),
        )
    
    @function_tool()
    async def update_reservation_time(
        self,
        time: Annotated[str, Field(description="The reservation time")],
        context: RunContext_T,
    ) -> str:
        """Called when the user provides their reservation time.
        Confirm the time with the user before calling the function."""
        userdata = context.userdata
        userdata.reservation_time = time
        return f"The reservation time updated to {time}"
    
    @function_tool()
    async def confirm_reservation(self, context: RunContext_T) -> str | tuple[Agent, str]:
        """Called when the user confirms the reservation."""
        userdata = context.userdata
        if not userdata.customer_name or not userdata.customer_phone:
            return "Please provide your name and phone first."
        
        if not userdata.reservation_time:
            return "Please provide reservation time first."
        
        return await self._transfer_to_agent("greeter", context)


# Takeaway agent
class Takeaway(BaseAgent):
    def __init__(self, menu: str) -> None:
        super().__init__(
            instructions=(
                f"You are a takeaway agent that takes orders from the customer."
                f"Our menu is: {menu}"
                "Clarify special requests and confirm the order from the customer."
            ),
            tools=[to_greeter],
            tts = cartesia.TTS(model="sonic3", voice=voices["takeaway"]),
        )

    @function_tool()
    async def update_order(
        self,
        items: Annotated[list[str], Field(description="The items of the full order")],
        context: RunContext_T,
    ) -> str:
        """Called when the user creates or updates their order."""
        userdata = context.userdata
        userdata.order = items
        return f"The order is updated to {items}"
    
    @function_tool()
    async def to_checkout(self, context: RunContext_T) -> str | tuple[Agent, str]:
        """Called when the user confirms the order."""
        userdata = context.userdata
        if not userdata.order:
            return "No takeaway order found. Please make an order first."
        
        return await self._transfer_to_agent("checkout", context)


# Checkout agent
class Checkout(BaseAgent):
    def __init__(self, menu: str) -> None:
        super().__init__(
            instructions=(
                f"You are a checkout agent at a restaurant. The menu is: {menu}\n"
                "You are responsible for confirming the expense of the "
                "order and then collecting customer's name, phone number and credit card "
                "information, including the card number, expiry date, and CVV step by step."
            ),
            tools=[update_name, update_phone, to_greeter],
            tts=cartesia.TTS(model="sonic-3", voice=voices["checkout"]), 
        )
    
    @function_tool()
    async def confirm_expense(
        self,
        expense: Annotated[float, Field(description="The expense of the order")],
        context: RunContext_T,
    ) -> str:
        """Called when the user confirms the expense."""
        userdata = context.userdata
        userdata.expense = expense
        return f"The expense is confirmed to be {expense}"
    
    @function_tool()
    async def update_credit_card(
        self,
        number: Annotated[str, Field(description="The credit card number")],
        expiry: Annotated[str, Field(description="The expiry date of the credit card")],
        cvv: Annotated[str, Field(description="The CVV of the credit card")],
        context: RunContext_T,
    ) -> str:
        """Called when the user provides their credit card number, expiry date, and CVV.
        Confirm the spelling with the user before calling the function."""
        userdata = context.userdata
        userdata.customer_credit_card = number
        userdata.customer_credit_card_expiry = expiry
        userdata.customer_credit_card_cvv = cvv
        return f"The credit card number is update to {number}"
    
    @function_tool()
    async def confirm_checkout(self, context: RunContext_T) -> str | tuple[Agent, str]:
        """Called when the user confirms the checkout."""
        userdata = context.userdata
        if not userdata.expense:
            return "Please confirm the expense first."
        
        if (
            not userdata.customer_credit_card
            or not userdata.customer_credit_card_expiry
            or not userdata.customer_credit_card_cvv
        ):
            return "Please provide the credit card information first."
        
        userdata.checked_out = True
        return await to_greeter(context)
    
    @function_tool()
    async def to_takeaway(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the user wants to update their order."""
        return await self._transfer_to_agent("takeaway", context)


server = AgentServer()

@server.rtc_session()
async def entrypoint(ctx: JobContext):
    menu = "Pizza: $10, Salad: $5, Ice Cream: $3, Coffee: $2"
    userdata = UserData()
    userdata.agents.update(
        {
            "greeter": Greeter(menu),
            "reservation": Reservation(),
            "takeaway": Takeaway(menu),
            "checkout": Checkout(menu)
        }
    )
    session = AgentSession[UserData](
        userdata=userdata,
        stt=openai.STT(model="gpt-4o-mini-transcribe"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(model="sonic-3"),
        vad=silero.VAD.load(),
        max_tool_steps=5,
    )

    await session.start(
        agent=userdata.agents["greeter"],
        room=ctx.room,
    )

if __name__ == "__main__":
    cli.run_app(server)

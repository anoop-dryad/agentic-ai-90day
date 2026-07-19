from pathlib import Path
import logging
import os
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------- logging ----------
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
log = logging.getLogger("agent")
log.setLevel(logging.DEBUG)
if not log.handlers:
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
    log.addHandler(console)


class AgentState(TypedDict):
    messages:Annotated[list[BaseMessage], add_messages]
    request_type:str
    awaiting_confirmation:bool

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0
)

def extract_text(message) -> str:
    if isinstance(message.content, str):
        return message.content
    return "".join(
        part.get("text", "")
        for part in message.content
        if isinstance(part, dict) and part.get("type") == "text"
    )

def classifier_node(state: AgentState) -> dict:
    """Classify the user's most recent message into a request type."""
    # Find the last user message
    last_user_msg = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    if not last_user_msg:
        return {"request_type": "chit_chat"}

    classification_prompt = (
        "Classify the following user message into ONE of these categories. "
        "Respond with ONLY the category name, nothing else.\n\n"
        "Categories:\n"
        "  time_query        - asking about current time in a city\n"
        "  coin_flip         - asking to flip a coin\n"
        "  destructive       - asking to delete, remove, or destroy something\n"
        "  ambiguous         - unclear what they want\n"
        "  chit_chat         - conversational, no tool needed\n\n"
        f"Message: {last_user_msg.content}\n\n"
        "Category:"
    )

    resp = model.invoke([HumanMessage(content=classification_prompt)])
    category = extract_text(resp).strip().lower().split()[0]

    # Constrain to known categories
    valid = {"time_query", "coin_flip", "destructive", "ambiguous", "chit_chat"}
    if category not in valid:
        category = "ambiguous"

    log.info(f"classify: '{last_user_msg.content[:50]}' → {category}")
    return {"request_type": category}
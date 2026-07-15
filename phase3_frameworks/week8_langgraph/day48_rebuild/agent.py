import os
import logging
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.utils.uuid import uuid7
from langgraph.prebuilt import ToolNode
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

from tools import get_current_time, flip_a_coin, broken_tool

# ---------- logging (same shape as Day 40) ----------
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

log = logging.getLogger("agent")
log.setLevel(logging.DEBUG)

if not log.handlers:
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))

    fh = logging.FileHandler(LOG_DIR / "agent.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    log.addHandler(console)
    log.addHandler(fh)

# ---------- middleware: catch tool exceptions ----------
@wrap_tool_call
def handle_tool_errors(request, handler):
    """Convert any tool exception into a ToolMessage the model can adapt to."""
    try:
        return handler(request)
    except Exception as e:
        log.warning(f"tool {request.tool_call['name']} crashed: {type(e).__name__}: {e}")
        return ToolMessage(
            content=f"Tool error: {type(e).__name__}: {e}. Please try a different approach.",
            tool_call_id=request.tool_call["id"],
        )


# ---------- model ----------
model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,
)


# ---------- agent ----------
SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "For any question about current time in a city, you MUST call get_current_time — "
    "even if the city sounds unusual. Let the tool decide. "
    "For coin flip requests, you MUST call flip_a_coin. "
    "If a tool returns an error with 'known_cities', offer them as alternatives. "
    "If the user says 'test failure', use broken_tool."
)

agent = create_agent(
    model=model,
    tools=[get_current_time, flip_a_coin, broken_tool],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),   # 🆕 conversation memory
    middleware=[handle_tool_errors]
)

# ---------- helper: get plain text from the last message ----------
def extract_text(message) -> str:
    """Handle both string and list-of-parts content shapes."""
    if isinstance(message.content, str):
        return message.content
    return "".join(
        part.get("text", "")
        for part in message.content
        if isinstance(part, dict) and part.get("type") == "text"
    )



# ---------- run ----------
if __name__ == "__main__":
    import time as _time

    # A single thread_id ties multiple turns into ONE conversation
    thread_id = str(uuid7())
    config = {"configurable": {"thread_id": thread_id}}

    questions = [
        "What time is it in Berlin?",
        "My favorite pastime is bouldering.",
        "I have a cat named Mochi.",
        "test failure",
        "And what time is it in Tokyo?",
        "What did I tell you my favorite pastime was?",   # tests memory
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n─── turn {i} ────────────────────────────────")
        print(f"🧑 {q}")
        log.info(f"turn {i}: {q}")

        result = agent.invoke(
            {"messages": [("user", q)]}, # type: ignore
            config=config,   # ← same config every turn = same conversation # type: ignore
        )
        answer = extract_text(result["messages"][-1])
        print(f"🤖 {answer}")

        if i < len(questions):
            _time.sleep(10)   # respect free-tier rate limit
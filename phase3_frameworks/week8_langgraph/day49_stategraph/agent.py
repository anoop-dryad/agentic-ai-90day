import os
import logging
from pathlib import Path
from typing import TypedDict, Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.utils.uuid import uuid7

from tools import get_current_time, flip_a_coin, broken_tool



# ---------- logging ----------
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


# ---------- state ----------
#LangGraph state must be a typed dict (or Pydantic model). 
# The framework needs a schema to know what each node can read and write.
class AgentState(TypedDict):
    """The data that flows through the graph."""
    messages: Annotated[list[BaseMessage], add_messages] #reducer, When multiple nodes update messages, add_messages says "append, don't replace


# ---------- model + tools ----------
model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,
)

tools = [get_current_time, flip_a_coin, broken_tool]
#it takes your tools' docstrings and type hints, 
# generates function-call schemas, and attaches them to the model.
model_with_tools = model.bind_tools(tools)   # tells the model what tools exist

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "For any question about current time in a city, you MUST call get_current_time — "
    "even if the city sounds unusual. Let the tool decide. "
    "For coin flip requests, you MUST call flip_a_coin. "
    "If a tool returns an error with 'known_cities', offer them as alternatives. "
    "If the user says 'test failure', use broken_tool."
)

# ---------- node: agent (calls the LLM) ----------
def agent_node(state: AgentState) -> dict:
    """Call the model with the current message history."""
    messages = state["messages"]

    # Prepend system prompt if it's not already there
    if not messages or messages[0].type != "system":
        from langchain_core.messages import SystemMessage
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    log.info(f"agent node: calling LLM with {len(messages)} messages")
    response = model_with_tools.invoke(messages)

    # Return only what changes — the framework merges via add_messages
    return {"messages": [response]}

# ---------- node: tools (executes tool calls) ----------
# ToolNode wraps our tools and handles the calling; we wrap it to add error handling.
_tool_node = ToolNode(tools)

def tools_node(state: AgentState) -> dict:
    """Execute any tool calls in the last message, catching exceptions."""
    last_message = state["messages"][-1]
    log.info(f"tools node: {len(last_message.tool_calls)} tool call(s)") # type: ignore

    try:
        result = _tool_node.invoke(state)
        return result
    except Exception as e:
        # Convert any tool crash into ToolMessages the LLM can see
        log.warning(f"tool crashed: {type(e).__name__}: {e}")
        error_msgs = [
            ToolMessage(
                content=f"Tool error: {type(e).__name__}: {e}",
                tool_call_id=call["id"],
            )
            for call in last_message.tool_calls # type: ignore
        ]
        return {"messages": error_msgs}
    
# ---------- routing: what happens after the agent node ----------
def route_after_agent(state: AgentState) -> str:
    """Decide the next node based on whether the LLM requested tools."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls: # type: ignore
        log.debug("routing: agent → tools")
        return "tools"
    log.debug("routing: agent → END")
    return END


# ---------- build the graph ----------
builder = StateGraph(AgentState)

# Add nodes
builder.add_node("agent", agent_node)
builder.add_node("tools", tools_node)

# Add edges
builder.add_edge(START, "agent")                          # start → agent
builder.add_conditional_edges("agent", route_after_agent)  # agent → tools OR END
builder.add_edge("tools", "agent")                        # tools → agent (loop back)


# Compile with a checkpointer for conversation memory
graph = builder.compile(checkpointer=InMemorySaver())


# ---------- helper ----------
def extract_text(message) -> str:
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

    thread_id = str(uuid7())
    config = {"configurable": {"thread_id": thread_id}}

    questions = [
        "What time is it in Berlin?",
        "My favorite pastime is bouldering.",
        "test failure",
        "And what time is it in Tokyo?",
        "What did I tell you my favorite pastime was?",
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n─── turn {i} ────────────────────────────────")
        print(f"🧑 {q}")

        result = graph.invoke(
            {"messages": [("user", q)]}, # type: ignore
            config={**config, "recursion_limit": 15}, # type: ignore
        )
        answer = extract_text(result["messages"][-1])
        print(f"🤖 {answer}")

        if i < len(questions):
            _time.sleep(3)
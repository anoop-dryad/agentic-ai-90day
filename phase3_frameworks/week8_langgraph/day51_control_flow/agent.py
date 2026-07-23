from pathlib import Path
import logging
import os
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import get_current_time, flip_a_coin,broken_tool, delete_user_data
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.utils.uuid import uuid7

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

tools=[get_current_time, flip_a_coin, broken_tool, delete_user_data]
model_with_tools=model.bind_tools(tools=tools)

def extract_text(message) -> str:
    if isinstance(message.content, str):
        return message.content
    return "".join(
        part.get("text", "")
        for part in message.content
        if isinstance(part, dict) and part.get("type") == "text"
    )

def classify_node(state: AgentState) -> dict:
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

def clarify_node(state: AgentState) -> dict:
    """The request was ambiguous. Ask for clarification without calling tools."""
    log.info("clarify: request was ambiguous")

    clarification = AIMessage(
        content=(
            "I'm not sure what you'd like me to do. I can help with:\n"
            "- Checking the current time in a city\n"
            "- Flipping a coin\n"
            "- Deleting user accounts (with confirmation)\n\n"
            "Could you rephrase what you need?"
        )
    )
    return{"messages":[clarification]}

CONFIRMATION_PHRASES = {"yes", "confirm", "do it", "proceed", "y"}

def confirmation_gate_node(state:AgentState) -> dict:
    """Ask for explicit confirmation before running destructive tools."""

    if state.get("awaiting_confirmation"):
        last_user_msg = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None,
        )

        if last_user_msg and any(
            phrase in last_user_msg.content.lower() for phrase in CONFIRMATION_PHRASES # type: ignore
        ):
            log.info("confirmation_gate: confirmed — proceeding")
            return {"awaiting_confirmation": False}
        else:
            log.info("confirmation_gate: not confirmed — cancelling")
            return {
                "awaiting_confirmation": False,
                "messages": [AIMessage(content="Cancelled. No changes were made.")],
                "request_type": "cancelled",
            }
    
     # First time seeing this destructive request — ask for confirmation
    log.info("confirmation_gate: asking for confirmation")
    return {
        "awaiting_confirmation": True,
        "messages": [AIMessage(
            content="This action is irreversible. Type 'yes' to confirm, or anything else to cancel."
        )],
    }

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the tools available for time queries, "
    "coin flips, and user deletions. Do not answer time or deletion questions "
    "from your own knowledge — always call the tool."
)

def agent_node(state: AgentState) -> dict:
    messages = state["messages"]
    context = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
    log.info(f"agent: LLM call with {len(messages)} messages")
    response = model_with_tools.invoke(context)
    return {"messages": [response]}

_tool_node = ToolNode(tools)

def tools_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    log.info(f"tools: {len(last_message.tool_calls)} tool call(s)") # type: ignore
    try:
        return _tool_node.invoke(state)
    except Exception as e:
        log.warning(f"tool crashed: {type(e).__name__}: {e}")
        return {"messages": [
            ToolMessage(
                content=f"Tool error: {type(e).__name__}: {e}",
                tool_call_id=call["id"],
            )
            for call in last_message.tool_calls # type: ignore
        ]}
    

def route_after_classify(state:AgentState) ->str:
    """After classification, decide where to go."""
    rt = state["request_type"]

    # If we're mid-confirmation flow, always run the gate
    if state.get("awaiting_confirmation"):
        return "confirmation_gate"
    
    if rt == "ambiguous":
        return "clarify"
    if rt == "destructive":
        return "confirmation_gate"
    # time_query, coin_flip, chit_chat all go to the normal agent
    return "agent"

def route_after_gate(state:AgentState) -> str:
    """After the gate, either proceed or exit."""
    if state.get("request_type") == "cancelled":
        return END
    if state.get("awaiting_confirmation"):
        return END   # waiting for user's next turn
    return "agent"   # confirmed, run the tool

def route_after_agent(state:AgentState) -> str:
    """Same as Day 49: tools if requested, else end."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls: # type: ignore
        return "tools"
    return END

builder = StateGraph(AgentState)

builder.add_node("classify", classify_node)
builder.add_node("clarify", clarify_node)
builder.add_node("confirmation_gate", confirmation_gate_node)
builder.add_node("agent", agent_node)
builder.add_node("tools", tools_node)

builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route_after_classify)
builder.add_edge("clarify", END)
builder.add_conditional_edges("confirmation_gate", route_after_gate)
builder.add_conditional_edges("agent", route_after_agent)
builder.add_edge("tools", "agent")

graph = builder.compile(checkpointer=InMemorySaver())

if __name__ == "__main__":
    import time as _time

    thread_id = str(uuid7())
    config = {"configurable": {"thread_id": thread_id}}

    tests = [
        # (label, user message)
        ("normal time query",           "What time is it in Berlin?"),
        ("chit chat",                    "That's cool, thanks."),
        ("ambiguous",                    "Do the thing."),
        ("destructive → first ask",      "Delete user alice."),
        ("destructive → confirm",        "yes"),
        ("destructive → cancel path",    "Delete user bob."),
        ("destructive → user cancels",   "actually no, don't"),
    ]

    for i, (label, q) in enumerate(tests, 1):
        print(f"\n─── turn {i}: {label} ────────────────────────────────")
        print(f"🧑 {q}")

        result = graph.invoke(
            {"messages": [("user", q)]}, # type: ignore
            config={**config, "recursion_limit": 15}, # type: ignore
        )
        answer = extract_text(result["messages"][-1])
        print(f"🤖 {answer}")

        if i < len(tests):
            _time.sleep(3)
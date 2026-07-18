import os
import logging
from pathlib import Path
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode
from tools import get_current_time, flip_a_coin,broken_tool
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

    fh = logging.FileHandler(LOG_DIR / "agent.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    log.addHandler(console)
    log.addHandler(fh)



class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0
)

tools=[get_current_time, flip_a_coin, broken_tool]
model_with_tools=model.bind_tools(tools=tools)

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "For any question about current time in a city, you MUST call get_current_time — "
    "even if the city sounds unusual. Let the tool decide. "
    "For coin flip requests, you MUST call flip_a_coin. "
    "If a tool returns an error with 'known_cities', offer them as alternatives. "
    "If the user says 'test failure', use broken_tool."
)

SUMMARY_TRIGGER = 10   # summarize when messages exceed this
KEEP_RECENT = 4        # keep the last N messages verbatim

def agent_node(state: AgentState) ->dict:
    messages = state["messages"]
    summary = state.get("summary", "")

    context = [SystemMessage(content=SYSTEM_PROMPT)]
    if summary:
        context.append(SystemMessage(
            content=f"[Earlier conversation summary]\n{summary}"
        ))
    context.extend(messages) # type: ignore
    log.info(f"agent node: {len(messages)} messages + summary({len(summary)} chars)")
    response = model_with_tools.invoke(context)

    return {"messages": [response]}

_tool_node = ToolNode(tools=tools)

def tools_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    try:
        result = _tool_node.invoke(state)
        return result
    except Exception as e:
        error_msgs = [
            ToolMessage(
                content=f"Tool error: {type(e).__name__}: {e}",
                tool_call_id=call["id"],
            )
            for call in last_message.tool_calls # type: ignore
        ]
        return {"messages": error_msgs}
    
def summarize_node(state: AgentState) -> dict:
    """Compress old messages into a summary, remove them from history."""
    messages = state["messages"]
    existing_summary = state.get("summary", "")
    to_summarize = messages[:-KEEP_RECENT]
    to_remove_ids = [m.id for m in to_summarize if m.id is not None]

    # Build the summarization prompt
    convo_text = "\n".join(
        f"{m.type}: {getattr(m, 'content', '')}"
        for m in to_summarize
        if getattr(m, "content", None)
    )

    prompt = (
        f"You previously summarized part of a conversation as:\n"
        f"{existing_summary}\n\n"
        f"Extend that summary with these additional turns. "
        f"Keep it under 100 words. Preserve user preferences, key facts, "
        f"and decisions. Drop pleasantries and exact tool arguments.\n\n"
        f"Additional turns:\n{convo_text}\n\nUpdated summary:"
    )

    log.info(f"📦 summarizing {len(to_summarize)} messages...")
    resp = model.invoke([HumanMessage(content=prompt)])
    new_summary = extract_text(resp).strip()   # 🆕 use extract_text instead of str()
    # Strip the "Updated summary:" prefix if the LLM includes it
    for prefix in ["Updated summary:", "Summary:", "SUMMARY:"]:
        if new_summary.lower().startswith(prefix.lower()):
            new_summary = new_summary[len(prefix):].strip()
    log.info(f"📦 summary: {new_summary[:80]}...")

    return {
        "summary": new_summary,
        "messages": [RemoveMessage(id=mid) for mid in to_remove_ids],
    }

def route_after_agent(state: AgentState) -> str:
    """Decide the next node based on whether the LLM requested tools."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls: # type: ignore
        log.debug("routing: agent → tools")
        return "tools"
    log.debug("routing: agent → END")
    return END

def route_after_tools(state: AgentState) -> str:
    """After tools run, decide: summarize first, or go back to agent?"""
    if len(state["messages"]) > SUMMARY_TRIGGER:
        log.debug(f"routing: tools → summarize (history has {len(state['messages'])} messages)")
        return "summarize"
    return "agent"

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools",tools_node)
builder.add_node("summarize",summarize_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route_after_agent)
builder.add_conditional_edges("tools", route_after_tools)
builder.add_edge("summarize", "agent")

graph = builder.compile(checkpointer=InMemorySaver())

def extract_text(message) -> str:
    if isinstance(message.content, str):
        return message.content
    return "".join(
        part.get("text", "")
        for part in message.content
        if isinstance(part, dict) and part.get("type") == "text"
    )

if __name__ == "__main__":
    import time as _time

    thread_id = str(uuid7())
    config = {"configurable": {"thread_id": thread_id}}

    questions = [
        "What time is it in Berlin?",
        "My favorite pastime is bouldering.",
        "I have a cat named Mochi.",
        "What time is it in Tokyo?",
        "Flip a coin.",
        "What time is it in London?",
        "Flip another coin.",
        "What did I tell you my favorite pastime was?",     # test after summarization
        "What was my cat's name?",                          # test after summarization
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

        # Show the internal state after each turn
        state = graph.get_state(config).values # type: ignore
        n_msg = len(state.get("messages", []))
        has_summary = bool(state.get("summary"))
        print(f"   📊 state: {n_msg} messages, summary={'YES' if has_summary else 'no'}")

        if i < len(questions):
            _time.sleep(3)
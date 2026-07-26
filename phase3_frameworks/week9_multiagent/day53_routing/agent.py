"""Day 53 — router that dispatches to specialist agents."""

import os
import logging
from pathlib import Path
from typing import TypedDict, Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.utils.uuid import uuid7

from tools import BILLING_TOOLS, TECH_TOOLS


# ---------- logging ----------
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
log = logging.getLogger("router")
log.setLevel(logging.INFO)
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
    log.addHandler(h)


# ---------- state ----------
class RouterState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    route: str          # which specialist handled this


# ---------- shared model ----------
model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,
)


def extract_text(message) -> str:
    if isinstance(message.content, str):
        return message.content
    return "".join(
        p.get("text", "") for p in message.content
        if isinstance(p, dict) and p.get("type") == "text"
    )


# ---------- the router ----------
def router_node(state: RouterState) -> dict:
    """One cheap LLM call to decide which specialist handles this."""
    last_user = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
    )
    if not last_user:
        return {"route": "general"}

    prompt = (
        "Route this customer message to ONE department. "
        "Answer with only the department name.\n\n"
        "Departments:\n"
        "  billing  - invoices, payments, refunds, account balance, pricing\n"
        "  tech     - errors, bugs, outages, API issues, service status\n"
        "  general  - anything else, greetings, unclear requests\n\n"
        f"Message: {last_user.content}\n\nDepartment:"
    )
    resp = model.invoke([HumanMessage(content=prompt)])
    route = extract_text(resp).strip().lower().split()[0]

    if route not in {"billing", "tech", "general"}:
        route = "general"

    log.info(f"router: → {route}")
    return {"route": route}

def make_specialist(name: str, system_prompt: str, tools: list):
    """Build an agent node + tools node pair for one specialty."""
    bound_model = model.bind_tools(tools) if tools else model
    tool_node = ToolNode(tools) if tools else None

    def agent_fn(state: RouterState) -> dict:
        context = [SystemMessage(content=system_prompt)] + list(state["messages"])
        log.info(f"{name}: LLM call ({len(state['messages'])} msgs, {len(tools)} tools)")
        return {"messages": [bound_model.invoke(context)]}

    def tools_fn(state: RouterState) -> dict:
        last = state["messages"][-1]
        log.info(f"{name}: running {len(last.tool_calls)} tool call(s)") # type: ignore
        try:
            return tool_node.invoke(state) # type: ignore
        except Exception as e:
            log.warning(f"{name}: tool crashed: {e}")
            return {"messages": [
                ToolMessage(content=f"Tool error: {e}", tool_call_id=c["id"])
                for c in last.tool_calls # type: ignore
            ]}

    return agent_fn, tools_fn


BILLING_PROMPT = (
    "You are a billing specialist. You handle invoices, payments, and account balances. "
    "Always call a tool to look up real data — never guess amounts or invoice statuses. "
    "If asked about technical errors or outages, say that's outside your department."
)

TECH_PROMPT = (
    "You are a technical support specialist. You handle error codes, bugs, and service status. "
    "Always call a tool to look up error meanings — never guess. "
    "If asked about billing or payments, say that's outside your department."
)

GENERAL_PROMPT = (
    "You are a friendly customer service assistant. You have no tools. "
    "Answer greetings and general questions conversationally. "
    "For billing or technical issues, tell the user you'll route them to the right team."
)

billing_agent, billing_tools_node = make_specialist("billing", BILLING_PROMPT, BILLING_TOOLS)
tech_agent, tech_tools_node = make_specialist("tech", TECH_PROMPT, TECH_TOOLS)
general_agent, _ = make_specialist("general", GENERAL_PROMPT, [])

def route_from_router(state: RouterState) -> str:
    return {"billing": "billing_agent",
            "tech": "tech_agent",
            "general": "general_agent"}[state["route"]]


def make_agent_router(tools_node_name: str):
    """After a specialist speaks: run its tools, or finish."""
    def route(state: RouterState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return tools_node_name
        return END
    return route


builder = StateGraph(RouterState)

builder.add_node("router", router_node)
builder.add_node("billing_agent", billing_agent)
builder.add_node("billing_tools", billing_tools_node)
builder.add_node("tech_agent", tech_agent)
builder.add_node("tech_tools", tech_tools_node)
builder.add_node("general_agent", general_agent)

builder.add_edge(START, "router")
builder.add_conditional_edges("router", route_from_router)

builder.add_conditional_edges("billing_agent", make_agent_router("billing_tools"))
builder.add_edge("billing_tools", "billing_agent")

builder.add_conditional_edges("tech_agent", make_agent_router("tech_tools"))
builder.add_edge("tech_tools", "tech_agent")

builder.add_edge("general_agent", END)

graph = builder.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    import time as _time

    config = {"configurable": {"thread_id": str(uuid7())}}

    tests = [
        "Hi there!",
        "What's the status of invoice INV-002?",
        "I'm getting error E429, what does that mean?",
        "How much do I owe on account CUST-42?",
        "Is the API service up right now?",
        "What time is it in Berlin?",   # neither department — watch what happens
    ]

    for i, q in enumerate(tests, 1):
        print(f"\n─── turn {i} ────────────────────────────────")
        print(f"🧑 {q}")
        result = graph.invoke(
            {"messages": [("user", q)]}, # type: ignore
            config={**config, "recursion_limit": 12}, # type: ignore
        )
        print(f"🤖 {extract_text(result['messages'][-1])}")
        if i < len(tests):
            _time.sleep(3)
"""Day 54 — supervisor delegates to specialists, receives results, decides next."""

import os
import logging
import operator
from pathlib import Path
from typing import TypedDict, Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.utils.uuid import uuid7

from tools import BILLING_TOOLS, TECH_TOOLS

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
log = logging.getLogger("supervisor")
log.setLevel(logging.INFO)
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
    log.addHandler(h)

MAX_DELEGATIONS = 4   # safety cap — supervisor's version of MAX_ITERATIONS

class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    billing_messages: Annotated[list[BaseMessage], add_messages]
    tech_messages: Annotated[list[BaseMessage], add_messages]
    next_agent: str
    findings: Annotated[list[str], operator.add]
    consulted: Annotated[list[str], operator.add]
    delegation_count: Annotated[int, operator.add]

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

SUPERVISOR_PROMPT = """You are a support supervisor coordinating specialist teams.

  billing  - invoices, payments, refunds, account balances
  tech     - error codes, bugs, outages, API and service status

Read the user's request and the findings so far, then decide the NEXT step.

Respond with EXACTLY one word: billing, tech, or DONE

Rules:
- If a finding contains '[out of scope: X]', the request still needs the team that owns X.
- Only answer DONE when every part of the user's request has been addressed by the
  team that owns it.
"""

def supervisor_node(state: SupervisorState) -> dict:
    if state.get("delegation_count", 0) >= MAX_DELEGATIONS:
        log.warning("supervisor: hit delegation cap — forcing DONE")
        return {"next_agent": "DONE"}

    consulted = set(state.get("consulted", []))
    available = [s for s in ("billing", "tech") if s not in consulted]

    if not available:
        log.info("supervisor: all specialists consulted → DONE")
        return {"next_agent": "DONE"}

    user_msg = next(
        (m for m in state.get("messages", []) if isinstance(m, HumanMessage)), None
    )
    findings = state.get("findings", [])

    context = f"User request: {user_msg.content if user_msg else '(none)'}\n\n"
    if findings:
        context += "Information gathered so far:\n"
        context += "\n".join(f"- {f}" for f in findings) + "\n\n"
    else:
        context += "No information gathered yet.\n\n"
    context += f"Specialists still available: {', '.join(available)}\n"
    context += f"Next step ({' / '.join(available)} / DONE):"

    resp = model.invoke([
        SystemMessage(content=SUPERVISOR_PROMPT),
        HumanMessage(content=context),
    ])
    decision = extract_text(resp).strip().split()[0].lower()

    if decision not in available:
        decision = "DONE"

    log.info(f"supervisor: → {decision} (consulted: {consulted or 'none'})")
    return {"next_agent": "DONE" if decision == "done" else decision}

def make_specialist(name: str, channel: str, system_prompt: str, tools: list):
    bound = model.bind_tools(tools)
    tool_node = ToolNode(tools, messages_key=channel)

    def agent_fn(state: SupervisorState) -> dict:
        user_msg = next(
            (m for m in state.get("messages", []) if isinstance(m, HumanMessage)), None
        )
        own = state.get(channel, [])
        context = (
            [SystemMessage(content=system_prompt)]
            + ([user_msg] if user_msg else [])
            + list(own)
        )
        log.info(f"{name}: LLM call ({len(own)} own msgs)")
        return {channel: [bound.invoke(context)]}

    def tools_fn(state: SupervisorState) -> dict:
        own = state.get(channel, [])
        if not own:
            return {}
        last = own[-1]
        log.info(f"{name}: running {len(last.tool_calls)} tool call(s)")
        try:
            return tool_node.invoke(state)
        except Exception as e:
            log.warning(f"{name}: tool crashed: {e}")
            return {channel: [
                ToolMessage(content=f"Tool error: {e}", tool_call_id=c["id"])
                for c in last.tool_calls
            ]}

    def report_fn(state: SupervisorState) -> dict:
        own = state.get(channel, [])
        last_ai = next((m for m in reversed(own) if isinstance(m, AIMessage)), None)
        finding = f"[{name}] {extract_text(last_ai) if last_ai else 'no result'}"
        log.info(f"{name}: finding = {finding[:150]}")
        return {
            "findings": [finding],
            "consulted": [name],
            "delegation_count": 1,
        }

    return agent_fn, tools_fn, report_fn


BILLING_PROMPT = (
    "You are a billing specialist. You handle ONLY invoices, payments, and account balances. "
    "Always call a tool for real data — never guess.\n\n"
    "CRITICAL: If the request mentions anything outside billing — error codes, API status, "
    "outages, technical issues — you MUST NOT answer that part. State exactly: "
    "'[out of scope: <topic>]' and answer only the billing portion. "
    "Never speculate about technical matters, even if you think you know.\n\n"
    "Answer concisely — your response goes to a supervisor, not the customer."
)

TECH_PROMPT = (
    "You are a technical support specialist. You handle ONLY error codes, bugs, "
    "outages, and service status. Always call a tool for real data — never guess.\n\n"
    "CRITICAL: If the request mentions anything outside tech — invoices, payments, "
    "balances, customer IDs — you MUST NOT answer that part. State exactly: "
    "'[out of scope: <topic>]' and answer only the technical portion. "
    "Never speculate about billing matters, even if you think you know.\n\n"
    "Answer concisely — your response goes to a supervisor, not the customer."
)

billing_agent, billing_tools, billing_report = make_specialist(
    "billing", "billing_messages", BILLING_PROMPT, BILLING_TOOLS
)
tech_agent, tech_tools, tech_report = make_specialist(
    "tech", "tech_messages", TECH_PROMPT, TECH_TOOLS
)

FINAL_PROMPT = (
    "You are a customer support agent. Using the information gathered by "
    "specialist teams, write a single clear answer to the customer's question. "
    "Be concise and friendly. Do not mention internal teams or processes."
)


def final_answer_node(state: SupervisorState) -> dict:
    """Synthesize all findings into one customer-facing answer."""
    user_msg = next(
        (m for m in state["messages"] if isinstance(m, HumanMessage)), None
    )
    findings = state.get("findings", [])

    context = f"Customer asked: {user_msg.content if user_msg else '(none)'}\n\n"
    if findings:
        context += "Information from specialist teams:\n"
        context += "\n".join(findings)
    else:
        context += "(No specialist information available.)"

    log.info(f"final: synthesizing {len(findings)} finding(s)")
    resp = model.invoke([
        SystemMessage(content=FINAL_PROMPT),
        HumanMessage(content=context),
    ])
    return {"messages": [AIMessage(content=extract_text(resp))]}

def route_from_supervisor(state: SupervisorState) -> str: # type: ignore
    return {
        "billing": "billing_agent",
        "tech": "tech_agent",
        "DONE": "final_answer",
    }[state["next_agent"]]


def make_specialist_router(channel: str, tools_name: str, report_name: str):
    def route(state: SupervisorState) -> str:
        own = state.get(channel, [])
        if own and getattr(own[-1], "tool_calls", None):
            return tools_name
        return report_name
    return route


builder = StateGraph(SupervisorState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("billing_agent", billing_agent)
builder.add_node("billing_tools", billing_tools)
builder.add_node("billing_report", billing_report)
builder.add_node("tech_agent", tech_agent)
builder.add_node("tech_tools", tech_tools)
builder.add_node("tech_report", tech_report)
builder.add_node("final_answer", final_answer_node)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route_from_supervisor)
builder.add_edge("billing_tools", "billing_agent")
builder.add_conditional_edges(
    "billing_agent",
    make_specialist_router("billing_messages", "billing_tools", "billing_report"),
)
builder.add_edge("billing_report", "supervisor")      # ← report back

builder.add_conditional_edges(
    "tech_agent",
    make_specialist_router("tech_messages", "tech_tools", "tech_report"),
)
builder.add_edge("tech_tools", "tech_agent")
builder.add_edge("tech_report", "supervisor")         # ← report back

builder.add_edge("final_answer", END)

graph = builder.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    import time as _time

    tests = [
        "What's the status of invoice INV-002?",
        "I'm getting error E429 during checkout AND my invoice INV-002 shows unpaid. "
        "Are these related?",
        "Is the API up, and how much do I owe on CUST-42?",
    ]

    for i, q in enumerate(tests, 1):
        print(f"\n{'═' * 60}")
        print(f"🧑 {q}")
        print("─" * 60)

        config = {"configurable": {"thread_id": str(uuid7())}}
        result = graph.invoke(
            {"messages": [("user", q)], "findings": [], "delegation_count": 0}, # type: ignore
            config={**config, "recursion_limit": 25}, # type: ignore
        )
        print(f"\n🤖 {extract_text(result['messages'][-1])}")
        print(f"   📊 {result.get('delegation_count', 0)} delegations, "
              f"{len(result.get('findings', []))} findings")

        if i < len(tests):
            _time.sleep(10)
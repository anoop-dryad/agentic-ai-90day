import logging
from pathlib import Path
from typing import TypedDict, Annotated
import operator
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.utils.uuid import uuid7

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
log = logging.getLogger("reflect")
log.setLevel(logging.INFO)
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
    log.addHandler(h)

MAX_REVISIONS=3

class ReflectState(TypedDict):
    task: str                                       # what we're writing
    draft: str                                      # current version
    critique: str                                   # latest feedback
    approved: bool                                  # critic said it's good
    revision_count: Annotated[int, operator.add]
    history: Annotated[list[str], operator.add]     # every draft, for inspection

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.3,   # slight creativity for drafting
)

def extract_text(message) -> str:
    if isinstance(message.content, str):
        return message.content
    return "".join(
        p.get("text","") for p in message.content
        if isinstance(p, dict) and p.get("type")=="text"
    )

WRITER_PROMPT = (
    "You are a customer support writer. Write clear, warm, concise replies. "
    "Answer every part of the customer's question. No corporate filler. "
    "Output only the reply text — no preamble, no explanation."
)

def generate_node(state: ReflectState) -> dict:
    task = state["task"]
    draft = state.get("draft", "")
    critique = state.get("critique","")

    if draft and critique:
        prompt = (
            f"Task: {task}\n\n"
            f"Your previous draft:\n{draft}\n\n"
            f"Editor's feedback:\n{critique}\n\n"
            f"Rewrite the reply addressing every point of feedback. "
            f"Output only the revised reply."
        )
        log.info(f"generate: revising (revision {state.get('revision_count', 0) + 1})")
    else:
        prompt = f"Task: {task}\n\nWrite the reply."
        log.info("generate: first draft")

    resp = model.invoke([SystemMessage(content=WRITER_PROMPT), HumanMessage(content=prompt)])
    new_draft = extract_text(resp).strip()

    return {"draft": new_draft, "history": [new_draft]}


CRITIC_PROMPT = """You are a strict but fair editor reviewing customer support replies.

Evaluate against these criteria:
1. Does it answer EVERY part of the customer's question?
2. Is any factual claim unsupported by the information given?
3. Is the tone warm without being sycophantic?
4. Is it under 50 words?

Respond in EXACTLY this format:

VERDICT: APPROVED
or
VERDICT: REVISE
ISSUES:
- <specific, actionable issue>
- <specific, actionable issue>

Rules:
- Approve if all four criteria pass. Do not invent problems.
- If revising, each issue must name what to change and where.
- Never write the revision yourself. Only identify problems.
"""
def critic_node(state: ReflectState) ->dict:
    task = state["task"]
    draft = state["draft"]

    word_count = len(draft.split())
    if word_count > 50:
        excess = word_count - 50
        log.info(f"critique: REVISE (length check: {word_count} words, {excess} over)")
        return {
            "critique": (
                f"VERDICT: REVISE\nISSUES:\n"
                f"- Draft is {word_count} words. Limit is 50. "
                f"Cut at least {excess} words. Remove the least essential sentence entirely "
                f"rather than trimming each sentence slightly."
            ),
            "approved": False,
            "revision_count": 1,
        }

    resp = model.invoke([SystemMessage(content=CRITIC_PROMPT), HumanMessage(content=f"Original task:\n{task}\n\nDraft to review:\n{draft}")])
    text = extract_text(resp).strip()

    approved = text.upper().startswith("VERDICT: APPROVED")
    log.info(f"critique: {'APPROVED' if approved else 'REVISE'}")
    if not approved:
        log.info(f"critique: {text[:200]}")

    return {
        "critique": text,
        "approved": approved,
        "revision_count": 0 if approved else 1,
    }

def route_after_critique(state: ReflectState) -> str:
    if state.get("approved"):
        log.info("route: approved → END")
        return END
    if state.get("revision_count", 0) >= MAX_REVISIONS:
        log.warning(f"route: hit {MAX_REVISIONS} revisions → END (shipping as-is)")
        return END
    return "generate"

builder = StateGraph(ReflectState)
builder.add_node("generate", generate_node)
builder.add_node("critique", critic_node)

builder.add_edge(START, "generate")
builder.add_edge("generate", "critique")
builder.add_conditional_edges("critique", route_after_critique)

graph = builder.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    import time as _time

    tasks = [
        "Customer asks: 'My invoice INV-002 shows unpaid but I paid last week. "
        "What's going on?' Known facts: INV-002 is 149 EUR, marked unpaid, "
        "issued 2026-07-01. Payments can take 3 business days to reflect.",

        "Customer asks: 'Why does your API keep giving me E429?' "
        "Known facts: E429 means rate limit exceeded. Fix is to wait 60 seconds "
        "and retry with exponential backoff. Their plan allows 100 req/min.",

        "Customer asks: 'I've been charged twice for INV-002 and your support "
        "hasn't replied in 4 days. This is unacceptable — I want a refund and "
        "an explanation.' Known facts: INV-002 is 149 EUR, one payment received "
        "2026-07-03, a second pending authorization (not a charge) from "
        "2026-07-04 that will auto-release in 5 days. Support queue delay was "
        "caused by an incident, now resolved. No refund is owed since only one "
        "payment was captured.",
    ]

    for i, task in enumerate(tasks, 1):
        print(f"\n{'═' * 60}")
        print(f"TASK {i}")
        print("─" * 60)

        config = {"configurable": {"thread_id": str(uuid7())}}
        result = graph.invoke(
            {"task": task}, # type: ignore
            config={**config, "recursion_limit": 20}, # type: ignore
        )

        print(f"\n📝 FINAL DRAFT:\n{result['draft']}")
        print(f"\n   📊 {len(result.get('history', []))} draft(s), "
              f"approved={result.get('approved')}")

        if len(result.get("history", [])) > 1:
            print(f"\n   🔍 First draft was:\n   {result['history'][0][:200]}...")

        if i < len(tasks):
            _time.sleep(10)
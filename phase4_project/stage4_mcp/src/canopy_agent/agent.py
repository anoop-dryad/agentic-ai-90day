"""Stage 0 — walking skeleton: LangGraph agent over the real device backend."""

import os

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from canopy_agent.backend import (
    get_device,
    list_devices,
)
from canopy_agent.health import compute_health
from canopy_agent.rag import search_docs_gated

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,
)


@tool
def device_status(device_id: str) -> dict:
    """Get the current health of a device by its ID (e.g. 'dev-001').

    Returns verified device data AND computed health flags (is_stale,
    low_battery, healthy, concerns). If the backend could not be verified,
    returns verified=False and the assistant must escalate honestly.
    """
    result = get_device(device_id)

    # --- FAILURE cases first ---
    if not result.ok:
        # not found is a real fact, distinct from "couldn't verify"
        if result.data and result.data.get("not_found"):
            return {
                "verified": False,
                "not_found": True,
                "reason": result.reason,
            }
        # any other gate failure
        return {
            "verified": False,
            "reason": result.reason,
        }

    device = result.data
    if not device or "id" not in device:
        return {
            "verified": False,
            "reason": "backend returned ok but data was incomplete",
        }

    health = compute_health(device)  # ← the judgment, in Python

    return {
        "verified": True,
        "id": device["id"],
        "name": device["name"],
        "status_field": device["status"],
        "battery_pct": device["battery_pct"],
        "last_seen": device["last_seen"],
        "health": health,  # ← LLM describes these flags
    }


@tool
def all_devices() -> dict:
    """List all devices. Returns verified data or an un-verified signal."""
    result = list_devices()
    if result.ok:
        return {"verified": True, "data": result.data}
    return {"verified": False, "reason": result.reason}


@tool
def search_docs(question: str) -> dict:
    """Search Dryad Silvanet documentation for how-to, troubleshooting, and
    knowledge questions — e.g. 'what does Inactive status mean?', 'how do I
    fix connectivity issues?', 'what is calibration mode?'. Use this for
    procedural or conceptual questions, NOT for a specific device's live
    status. Returns grounded doc excerpts, or indicates no relevant docs were
    found — in which case do not invent an answer.
    """
    result = search_docs_gated(question=question)
    if not result:
        return {
            "grounded": False,
            "reason": result.reason,
        }

    return {
        "grounded": True,
        "excerpts": result.chunks,
    }


SYSTEM_PROMPT = (
    "You are a Dryad Silvanet support assistant.\n\n"
    "TOOL SELECTION:\n"
    "- For a specific device's LIVE status/health (is dev-X active, its energy, "
    "when last seen), use device_status.\n"
    "- For what things MEAN or HOW to do them (what Inactive means, calibration, "
    "connectivity troubleshooting, alert severity), use search_docs.\n"
    "- A question may need BOTH: check a device's live status, then explain what "
    "that status means from the docs.\n\n"
    "GROUNDING RULES:\n"
    "- device_status verified=False → never invent status; escalate honestly. "
    "not_found → device doesn't exist. auth_failed → trouble accessing backend.\n"
    "- search_docs grounded=False → say you don't have documentation on that and "
    "offer to escalate. NEVER answer from your own knowledge when grounded=False.\n"
    "- Report device HEALTH, not just the raw status (stale/low-energy = concern).\n"
    "- Answer ONLY from tool data — verified device data or retrieved docs."
)

agent = create_agent(
    model=model,
    tools=[device_status, all_devices, search_docs],
    system_prompt=SYSTEM_PROMPT,
)


def extract_text(message) -> str:
    if isinstance(message.content, str):
        return message.content
    return "".join(
        p.get("text", "")
        for p in message.content
        if isinstance(p, dict) and p.get("type") == "text"
    )


if __name__ == "__main__":
    questions = [
        "What's the status of device dev-001?",
        "Is dev-002 online?",
        "What about dev-999?",  # not in your seed data → not found
        "How's the battery on dev-002?",  # your seed has dev-002 at 12%
        "what does Inactive status mean?",
        "why is dev-003 inactive?",
        "what's the airspeed of a swallow?",
    ]
    for q in questions:
        print(f"\n{'─' * 60}\n🧑 {q}")
        result = agent.invoke({"messages": [("user", q)]})
        print(f"🤖 {extract_text(result['messages'][-1])}")

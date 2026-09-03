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


SYSTEM_PROMPT = (
    "You are a Dryad device support assistant. Answer using ONLY tool data.\n\n"
    "CRITICAL RULES:\n"
    "1. When verified=False, never invent a status. Say you couldn't verify and "
    "are escalating. If not_found=True, say the device doesn't exist.\n"
    "2. When verified=True, report the device's HEALTH, not just its status field.\n"
    "3. If health.healthy is False, LEAD with the concern(s) in health.concerns. "
    "Do not say a device is 'fine' or 'online' without qualification when there "
    "are concerns. A device whose status is 'online' but is_stale or low_battery "
    "is NOT fine — say so clearly and first.\n"
    "4. If health.healthy is True, you may confirm the device is operating normally.\n"
    "5. Be direct about what needs attention — this is safety equipment."
)

agent = create_agent(
    model=model,
    tools=[device_status, all_devices],
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
    ]
    for q in questions:
        print(f"\n{'─' * 60}\n🧑 {q}")
        result = agent.invoke({"messages": [("user", q)]})
        print(f"🤖 {extract_text(result['messages'][-1])}")

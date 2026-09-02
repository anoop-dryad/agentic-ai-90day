"""Stage 0 — walking skeleton: LangGraph agent over the real device backend."""

import os

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from canopy_agent.backend import (
    get_device,
    list_devices,
)

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,
)


@tool
def device_status(device_id: str) -> dict:
    """Get the current status of a device by its ID (e.g. 'dev-001').

    Returns the device's real status, battery, and last-seen time if the
    backend could be reached and returned valid data. If the device was
    not found or the backend could not be verified, returns an error the
    assistant must relay honestly — never guess a status.
    """
    result = get_device(device_id)
    if result.ok:
        d = result.data
        return {
            "verified": True,
            "id": d["id"],
            "name": d["name"],
            "status": d["status"],
            "battery_pct": d["battery_pct"],
            "last_seen": d["last_seen"],
        }
    # Gate failed — hand the LLM an explicit un-verified signal, not data.
    return {
        "verified": False,
        "reason": result.reason,
    }


@tool
def all_devices() -> dict:
    """List all devices. Returns verified data or an un-verified signal."""
    result = list_devices()
    if result.ok:
        return {"verified": True, "data": result.data}
    return {"verified": False, "reason": result.reason}


SYSTEM_PROMPT = (
    "You are a Dryad device support assistant. You answer questions about "
    "device status using ONLY data returned by your tools.\n\n"
    "CRITICAL RULES:\n"
    "1. When a tool returns verified=True, describe the real data accurately.\n"
    "2. When a tool returns verified=False, you MUST NOT invent or guess a "
    "device's status. Tell the user you could not verify the information and "
    "that you're escalating to a human specialist. State the reason if helpful.\n"
    "3. Never state a device is online, offline, or fine unless a tool verified it.\n"
    "4. If a device was not found, say so clearly — that is different from "
    "'could not check'."
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


# STAGE 0 — COMPLETE ✅
# Works:  4/4 correct on happy path; not-found ≠ couldn't-verify.
# Safe:   backend killed → 4/4 honest escalation, ZERO fabricated statuses.
#         dev-002 described in detail with backend up, refused to invent it
#         with backend down — the gate controls what the LLM sees.
# FINDING (Stage 1+): agent echoes status field, doesn't reason about health.
#         12% battery buried under "yes, online". Health should be headline.
# Thesis proven: structure enforces what the prompt requests.

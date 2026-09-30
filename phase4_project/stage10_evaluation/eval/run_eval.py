"""Deterministic agent evaluation runner.

Runs each labeled scenario through the REAL agent (over MCP), captures the
tool the agent chose and its answer, and checks them against expected behavior.
This is the reliable base layer — fast to judge (string/state checks), no judge
LLM. It DOES call the agent (which uses Gemini), so it's a manual pre-ship gate,
not a CI test."""

import asyncio

from canopy_agent.config import settings
from canopy_agent.mcp_client_agent import process_query
from eval.eval_set import EVAL_CASES
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


def _check_answer(answer: str, case: dict) -> list[str]:
    """Deterministic checks on the answer text. Returns list of failures."""
    a = answer.lower()
    failures = []

    # every must_contain_any group: at least one alternative must appear
    for group in case.get("must_contain_any", []):
        if not any(alt.lower() in a for alt in group):
            failures.append(f"missing any of {group}")

    # no forbidden phrase may appear
    for phrase in case.get("must_not_contain", []):
        if phrase.lower() in a:
            failures.append(f"contains forbidden '{phrase}'")

    return failures


def _tool_used(trace: list[dict], expected: str) -> bool:
    """Did the agent call the expected tool during this query?"""
    return any(e.get("tool") == expected for e in trace if e.get("step") == "tool_call")


async def _run_one(session, case: dict) -> dict:
    """Run one scenario through the real agent, evaluate its behavior."""
    answer, trace, _pending = await process_query(session, case["question"])

    failures = []

    # 1. right tool?
    expected_tool = case.get("expect_tool")
    if expected_tool and not _tool_used(trace, expected_tool):
        called = [e.get("tool") for e in trace if e.get("step") == "tool_call"]
        failures.append(f"expected tool '{expected_tool}', called {called}")

    # 2. answer checks (facts present, fabrications absent)
    failures.extend(_check_answer(answer, case))

    return {
        "id": case["id"],
        "passed": len(failures) == 0,
        "failures": failures,
        "answer": answer,
    }


async def main():
    async with (
        streamable_http_client(settings.MCP_SERVER_URL) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()

        results = []
        for case in EVAL_CASES:
            r = await _run_one(session, case)
            results.append(r)
            await asyncio.sleep(5)  # free-tier rate limit

    # ── scorecard ──
    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    print(f"\n{'═' * 64}")
    print(f"AGENT EVALUATION: {passed}/{total} passed")
    print("═" * 64)
    for r in results:
        icon = "✅" if r["passed"] else "❌"
        print(f"\n{icon} {r['id']}")
        if not r["passed"]:
            for f in r["failures"]:
                print(f"     └─ {f}")
            print(f"     answer: {r['answer'][:120]}")

    print(f"\n{'═' * 64}")
    if passed < total:
        print(
            f"⚠️  {total - passed} case(s) failed — investigate (test may be wrong too)."
        )
    else:
        print("✅ All behavioral checks passed.")


if __name__ == "__main__":
    asyncio.run(main())

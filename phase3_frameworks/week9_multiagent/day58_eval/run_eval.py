"""Run every case, check outcome + path + failure-mode avoidance."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "day57_flows"))

from support_flow import SupportFlow  # your Day 57 flow
from eval_cases import CASES

def evaluate_one(case: dict) -> dict:
    """Run one case, return a structured pass/fail with reasons."""
    flow = SupportFlow()
    flow.kickoff(inputs={"question": case["question"]})

    answer = (flow.state.answer or "").lower()
    category = flow.state.category
    gate = flow.state.backend_ok

    failures = []

    # Check 2: right path (category)
    if category != case["expect_category"]:
        failures.append(f"category: got '{category}', expected '{case['expect_category']}'")

    # Check 2: right path (gate)
    if gate != case["expect_gate"]:
        failures.append(f"gate: got {gate}, expected {case['expect_gate']}")

    # Check 1 & 3: facts present, hallucinations absent
    for phrase in case.get("must_contain", []):
        if phrase.lower() not in answer:
            failures.append(f"missing required: '{phrase}'")
    for phrase in case.get("must_not_contain", []):
        if phrase.lower() in answer:
            failures.append(f"contains forbidden: '{phrase}'")

    return {
        "id": case["id"],
        "passed": len(failures) == 0,
        "failures": failures,
        "answer": flow.state.answer,
    }

def main():
    results = [evaluate_one(c) for c in CASES]

    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    print(f"\n{'═' * 64}")
    print(f"EVAL RESULTS: {passed}/{total} passed")
    print("═" * 64)

    for r in results:
        icon = "✅" if r["passed"] else "❌"
        print(f"\n{icon} {r['id']}")
        if not r["passed"]:
            for f in r["failures"]:
                print(f"     └─ {f}")
            print(f"     answer: {r['answer'][:120]}...")

    print(f"\n{'═' * 64}")
    if passed < total:
        print(f"⚠️  {total - passed} case(s) failed. Do not ship.")
        sys.exit(1)
    else:
        print("✅ All cases passed.")

if __name__ == "__main__":
    main()
"""Evaluation cases. Each says: input, and what MUST be true of the result."""

# Each case specifies checkable facts, not an exact expected string.
CASES = [
    {
        "id": "billing_valid",
        "question": "What's the status of invoice INV-002?",
        "expect_category": "billing",
        "expect_gate": True,               # gate must PASS
        "must_contain": ["149", "unpaid"], # facts that must appear
        "must_not_contain": ["paid,", "escalat"],  # signs of wrong answer
    },
    {
        "id": "billing_missing",
        "question": "What about invoice INV-999?",
        "expect_category": "billing",
        "expect_gate": False,              # gate must FAIL (invoice doesn't exist)
        "must_contain": ["escalat"],       # must honestly escalate
        "must_not_contain": ["149", "unpaid", "paid"],  # must NOT invent a status
    },
    {
        "id": "tech_valid",
        "question": "What does error E429 mean?",
        "expect_category": "technical",
        "expect_gate": True,
        "must_contain": ["rate limit"],
        "must_not_contain": ["escalat"],
    },
    {
        "id": "tech_no_code",
        "question": "My dashboard looks weird.",
        "expect_category": "technical",
        "expect_gate": False,
        "must_contain": ["escalat"],
        "must_not_contain": ["E429", "rate limit"],  # must not invent an error
    },
    {
        "id": "billing_wording_variant",
        "question": "Can you check invoice INV-002 for me please?",
        "expect_category": "billing",
        "expect_gate": True,
        "must_contain": ["149"],
        "must_not_contain": ["escalat"],
    },
]
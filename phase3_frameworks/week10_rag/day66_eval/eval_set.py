"""Labeled evaluation set. Ground truth for retrieval + answers."""

# Each case: the question, facts the answer MUST contain, facts it must NOT,
# and a keyword that identifies the chunk that SHOULD be retrieved.
EVAL_SET = [
    {
        "id": "e429_meaning",
        "question": "What does error E429 mean?",
        "must_contain_any": [["rate limit", "too many requests"]],
        "must_not_contain": [],
        "relevant_chunk_marker": "E429",  # the right chunk contains this
    },
    {
        "id": "e507_fix",
        "question": "How do I fix E507?",
        "must_contain_any": [["upgrade", "delete"]],
        "must_not_contain": [],
        "relevant_chunk_marker": "E507",
    },
    {
        "id": "led_contrast",
        "question": "My sensor has no LED at all. Is that the same as a dead battery?",
        "must_contain_any": [["no power", "wiring"]],
        "must_not_contain": [],
        "relevant_chunk_marker": "No LED",
    },
    {
        "id": "reconnect_time",
        "question": "How long until a sensor reconnects on its own?",
        "must_contain_any": [["15 minutes"]],
        "must_not_contain": [],
        "relevant_chunk_marker": "15 minutes",
    },
    {
        "id": "alert_threshold",
        "question": "Why didn't I get an alert for a detection?",
        "must_contain_any": [["80%", "confidence"]],
        "must_not_contain": [],
        "relevant_chunk_marker": "80%",
    },
    {
        "id": "underwater_hallucination",
        "question": "Can I use my sensors underwater?",
        "must_contain_any": [["don't have", "escalate"]],  # must refuse
        "must_not_contain": ["waterproof", "yes you can", "meters"],  # must not invent
        "relevant_chunk_marker": None,  # nothing should confidently match
    },
    {
        "id": "warranty_partial",
        "question": "What's the warranty period on a sensor?",
        "must_contain_any": [["don't have", "escalate"]],
        "must_not_contain": ["year", "month", "warranty covers"],
        "relevant_chunk_marker": None,
    },
]

# Phase 2 — Hands-on Basics (Days 22–45)

Building agents from scratch in Python. No frameworks — just the Gemini SDK, `requests`, and Chroma. Each day extends the previous one, so the code progression itself tells the story.

**Goal by end of phase:** an agent with tools, memory, error recovery, structured logging, and both short-term (compression) and long-term (vector) memory — all built by hand, no framework abstractions.

---

## The Progression

### Week 4 — Setup + first LLM call (Days 22–28)

| Day | What was built |
|-----|----------------|
| 22  | Python 3.12 + VS Code installed |
| 23  | `agents-env` venv + package hygiene |
| 24  | Gemini API key from AI Studio |
| 25  | `.envrc` + `direnv` for secret loading |
| 26  | First `generate_content` call — 9 lines |
| 27  | Terminal chat loop with conversation history |
| 28  | Week review |

### Week 5 — First real agent (Days 29–35)

| Day | What was built |
|-----|----------------|
| 29  | Read Anthropic's *Building Effective Agents* (concept) |
| 30  | R-T-R-F system prompts in code (Python tutor + pirate variant) |
| 31  | Memory verification — logging the history growth and proving statelessness |
| 32  | Tool design refresher — the 6-step function-calling dance |
| 33  | First tool: `get_current_time` with mock data and errors-as-return-values |
| 34  | Wired the tool to the LLM — first working agent, plus fixed premature refusal |
| 35  | Week review |

### Week 6 — Multi-tool + ReAct + structure (Days 36–42)

| Day | What was built |
|-----|----------------|
| 36  | Second tool (`flip_a_coin`) — LLM has to choose between tools |
| 37  | Real API (TimeAPI.io after WorldTimeAPI died mid-lesson) |
| 38  | The while-loop — hardcoded two calls became a real ReAct loop |
| 39  | Error handling — every exception becomes structured data |
| 40  | Structured logging — `logging` module with console + file handlers |
| 41  | Refactor — one file split into `agent.py`, `config.py`, `logging_setup.py`, `tools/` |
| 42  | Week review |

### Week 7 — Memory engineering (Days 43–45)

| Day | What was built |
|-----|----------------|
| 43  | Instrumented context growth — watched tokens climb from 6 to 924 over 10 turns |
| 44  | Summarization — when history crosses 12 messages, LLM compresses old turns |
| 45  | Vector memory (Chroma) — semantic recall of past exchanges |

---

## Reading Order

If you're studying this repo, don't read all folders in order — the day-by-day copy-forward pattern means there's a lot of near-duplicate code. Instead, read these five folders in order:

1. **`day27_chat_loop/chat.py`** — the smallest useful thing: a stateful chat loop
2. **`day34_wire_tool/agent.py`** — the smallest working agent (one tool)
3. **`day38_react_loop/agent.py`** — the smallest ReAct loop
4. **`day41_refactor/`** — the same agent split into clean modules
5. **`day45_vector_memory/`** — the final Phase 2 form with both memory types

Each is a strict superset of the previous. If you can read all five, you can read any agent framework's source and recognize what it's doing under the abstractions.

---

## Architecture at Day 45 (final Phase 2 form)

```
phase2_basics/day45_vector_memory/
├── agent.py          # the loop
├── config.py         # model, system prompt, tool wiring
├── logging_setup.py  # console + file handlers
├── memory.py         # Chroma-backed long-term memory
└── tools/
    ├── __init__.py   # registry: FUNCTIONS + DECLARATIONS
    ├── time_tool.py
    ├── coin_tool.py
    └── broken_tool.py
```

Five files, each with a single reason to change:

- Change **what the agent does** → `agent.py`
- Add or remove a **tool** → drop a file in `tools/`, add two lines to `__init__.py`
- Change **the model, prompt, or defaults** → `config.py`
- Change **what you see when it runs** → `logging_setup.py`
- Change **how memory works** → `memory.py`

This is exactly what framework abstractions give you for free in Phase 3 — but building it by hand first means the abstractions won't feel magical.

---

## Real Bugs Fixed Along the Way

Documented because they were the actual learning:

- **Global-vs-local `history` bug (Day 36)** — a globally-defined history was carrying context between test cases. LLM answered questions with content from earlier prompts. Fixed by moving `history` inside `ask()`. Lesson: state placement = state semantics.
- **Premature refusal (Day 34)** — LLM refused to look up "Atlantis" from its own knowledge instead of calling the tool. Never fired the error-recovery path. Fixed with a `you MUST call the tool` clause. Lesson: tools compete with the LLM's own knowledge; the system prompt has to enforce authority.
- **WorldTimeAPI died mid-lesson (Day 37)** — `ConnectionResetError` on every call. Not a bug in my code; the vendor was down. Swapped to TimeAPI.io in 5 lines because the interface was decoupled from the backend. Lesson: clean seams pay their debt when the world breaks.
- **Retry-anything logic (Day 43)** — `call_llm` retried every exception the same way, including 4xx errors that would never succeed. Wasted quota and time. Fixed by distinguishing `ServerError` (retry) from `ClientError` (fail fast). Lesson: not every error deserves the same treatment.
- **Free-tier request rate, not context length (Day 43)** — expected to hit context limits first; hit request-per-minute quota first. On free tier, ReAct multiplies API calls fast (2–3 per user turn). Fixed with a `time.sleep(3)` between turns in the test.

---

## Design Decisions Worth Defending

- **Errors as data, not raised exceptions.** Tool crashes become `{"error": ...}` dicts the LLM reads and adapts to. Raised exceptions kill the loop mid-conversation.
- **Model name as a constant at the top of the file.** One place to swap, one place to grep, one place to A/B test.
- **`MAX_ITERATIONS` on every loop.** Removing it once "for testing" is how quota accidents happen.
- **File-level logging always on.** Console can be quiet; the file trail is what saves you when a user reports a bug from yesterday.
- **Refactored at Day 41, not later.** The 200-line single file was already producing bugs by adding a tool. Refactoring after production pain is more expensive than refactoring at the first symptom.

---

## What Phase 2 Does *Not* Include

Being explicit about scope so nothing is oversold:

- No evaluation harness (Phase 4 — Day 67 onward)
- No cost tracking beyond `approx_tokens()` / `real_tokens()` dashboards
- No safety layer or prompt-injection defense (Phase 4 — Day 71)
- No human-in-the-loop for risky actions (Phase 4 — Day 72)
- No multi-agent patterns (Phase 3 — Day 57)
- No UI — everything runs in the terminal
- No deployment — everything runs locally

Phase 3 and 4 fill these gaps. The point of Phase 2 was to build the primitives so I know what frameworks are actually abstracting.

---

## Running Any Day

Each folder is standalone:

```bash
cd phase2_basics/day45_vector_memory
python agent.py
```

Prerequisites: `agents-env` active, `GEMINI_API_KEY` loaded via `direnv`, dependencies installed. See root [README](../README.md) for setup.

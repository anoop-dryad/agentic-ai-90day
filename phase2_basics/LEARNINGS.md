# LEARNINGS.md

Lessons from building agents by hand — the ones I'd want to tell someone starting from scratch. Organized by theme, not chronology.

---

## About LLMs Themselves

**LLMs are next-token predictors with no memory and no ability to act.**
Everything else is engineering wrapped around this fact. I didn't fully believe this until Day 31 when I logged the history growing message by message and watched a `no_memory.py` version forget my favorite color one turn after I said it.

**"Confident" ≠ "correct."**
On Day 34, the LLM confidently refused to look up "Atlantis" from its own training data, bypassing my tool and never firing the error-recovery path I'd designed. It wasn't wrong that Atlantis is mythical. It was wrong to answer without asking. Fix: system prompt gives the tool authority. The pattern generalized — LLMs will trust themselves over your tool unless you explicitly tell them not to.

**Temperature 0 isn't fully deterministic.**
GPU floating-point + parallelism means even temp 0 has ~1–5% variance. Don't write tests that demand exact string matches across runs.

---

## About Tools

**A tool's description is the contract with the LLM.**
On Day 32, I wrote my first tool description with typos, no "when NOT to use it" section, and no documented return shape. Every one of those omissions became a bug on Day 34. The LLM reads the description literally — vague description, vague behavior.

**Fewer, specific tools > many, vague tools.**
Two overlapping tools cause the LLM to dither and pick differently across runs. On Day 36, adding a second tool made me feel the pressure that description quality had been hiding when there was only one.

**Return errors as data. Never raise.**
This one decision on Day 33 (`{"error": ...}` instead of `raise ValueError`) paid dividends every day after. The LLM reads errors, adapts, and offers alternatives. Raised exceptions kill the loop mid-conversation.

**A tool that can rewrite files, spend money, or delete records is not the same tool as one that reads data.**
Read-only tools are 10× safer than write tools. For learning: prefer read-only. For production: wrap risky writes in confirmation.

---

## About the Loop

**ReAct is a while-loop with a max-iteration cap. That's it.**
Frameworks make it look mysterious. It isn't. Day 38's code is ~15 lines and matches every framework's core execution model.

**The LLM signals "done" by emitting text instead of a function_call.**
Not by returning a special token. Not by calling a `finish` tool. Just by not asking for another tool. Your loop's exit condition is `if not part.function_call: return part.text`. This is the ReAct terminator.

**MAX_ITERATIONS is not paranoia. It's insurance.**
A tool with a poorly written description can cause the LLM to oscillate. Without a cap, that oscillation happens indefinitely until you or your quota noticed. Every loop I ever write again will have this cap.

**Two tools chained ≠ one tool with two args.**
On Day 38, asking "what time is it in Berlin AND Tokyo?" made the agent iterate twice — Berlin first, then Tokyo. The LLM decided each step. That's the difference between a hardcoded workflow and an actual agent.

---

## About Memory

**Memory is engineered, not inherited.**
The LLM is stateless. Every appearance of memory is you (or your framework) maintaining state, deciding what to include, and re-sending it. Day 31 was the moment this stopped being an abstract idea.

**A Python list works as memory until it doesn't.**
It's fine for demos and short conversations. On Day 43, I watched tokens climb from 6 to 924 over 10 turns. Extrapolate to 100 turns and it becomes expensive. To 1000 turns and it fails.

**Summarization is lossy; retrieval is not.**
Day 44 compressed old history into paragraphs — great when I needed the gist. Day 45 stored every exchange in Chroma and pulled back only the semantically relevant ones — better when I needed exact facts. Both together is what production apps do.

**Retrieval quality is the new bottleneck.**
Vector memory shifts the problem from "how do I store this?" to "how do I find the right things?" Bad top-K → bad answers. That's Phase 3's problem.

---

## About Error Handling

**Not every error deserves retry with backoff.**
On Day 43, my `call_llm` retried every exception the same way. Bad requests (4xx) will always fail; retrying wastes quota. Server errors (5xx) are transient; backoff helps. Distinguishing them made the retry logic honest.

**On free tiers, the enemy is requests-per-minute, not tokens.**
I expected Day 43 to hit context limits. Instead, hit the request-rate quota first because ReAct makes 2–3 API calls per user turn. Different bottleneck than the tutorials warn about.

**Log the real error, not just the exception type.**
`LLM error (ClientError)` tells you nothing. `LLM error (ClientError): 429 RESOURCE_EXHAUSTED` tells you exactly what to fix. Include the message every time.

---

## About Code Structure

**Refactor at the first symptom, not after the pain.**
By Day 41 the single-file agent was 200 lines and adding a tool required touching four sections of the same file. That's the moment to split — before bugs, not after them.

**Files should have one reason to change.**
Not one function, not one class — one reason to change. Adding a tool shouldn't touch logging. Changing the model shouldn't touch tools. This is what made adding new features fast for the rest of Phase 2.

**A registry pattern (`FUNCTIONS`, `DECLARATIONS`) scales; hardcoded switch statements don't.**
On Day 41 I created a dict of `{name: function}` and a list of declarations. Every tool added after was two lines in `__init__.py` and a new file. No agent.py changes.

---

## About Working with a Free-Tier LLM

**Model availability changes.**
On Day 26, `gemini-2.0-flash` gave me a quota error saying "limit: 0" — meaning I never had access to it. `gemini-3.1-flash-lite` worked. Always list models from code (`client.models.list()`) rather than trust a tutorial's model string.

**APIs die. Design for it.**
On Day 37, WorldTimeAPI collapsed mid-lesson. My tool's error handling caught it cleanly, and swapping to TimeAPI.io was a 5-line change because the interface was separate from the backend. Vendor lock-in is a design choice, and the design choice is "don't."

**`time.sleep(10)` between test loop iterations respects the request-per-minute cap.**
Not realistic user behavior, but essential for repeatable test runs on free tier.

---

## About the Journey Itself

**Concepts before code, in this order, was the right call.**
21 days of pure concepts felt slow. Then Phase 2 started and every code decision had a Phase 1 reason. If I'd started with tutorials, I'd have working demos and no understanding.

**"It ran" ≠ "it worked."**
On Day 39 my error-handling changes made the tests print `🤖 The current time in Berlin is 3:32 PM` but the trace showed the tool never ran. Only the logs would have told me. Observability is not optional; it's the difference between shipping a working thing and shipping a plausible-looking thing.

**Yak-shaving kills learners faster than confusion does.**
On Day 30 I lost 30 minutes to Pylance type warnings on library code I didn't own. Silencing the linter was the right call. Real engineering skill is knowing when to stop fighting tooling.

**Bugs I hit are worth more than lessons I read.**
The global-history contamination on Day 36, the premature refusal on Day 34, the dead API on Day 37 — those taught me more than any smooth day.

---

## What I'd Do Differently

- **Start with a `docs/` folder and a per-phase README from Day 1**, not build them as an afterthought.
- **Instrument token counts earlier** — the Day 43 growth would have been visible on Day 27 if I'd been logging tokens per turn.
- **Read Anthropic's _Building Effective Agents_ on Day 1**, not Day 29. It would have oriented every subsequent day.

---

## What I'm Watching For in Phase 3

Now that I've built the primitives, the question is: **what do frameworks actually give me beyond what I built?** Educated guesses:

- **Graph-based flow control** (LangGraph) — probably makes the loop declarative instead of imperative
- **Built-in observability + tracing** — the logging setup I hand-rolled on Day 40 is probably one line in most frameworks
- **Multi-agent orchestration** — I have no primitives for this yet
- **Streaming responses** — I've been printing final answers only; users expect token-by-token
- **Tool schema generation from Python type hints** — instead of hand-writing FunctionDeclarations

Whichever of these frameworks give me for free, I'll know to appreciate. Whichever they get wrong, I'll notice — because I've done it by hand.

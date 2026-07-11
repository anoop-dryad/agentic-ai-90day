# Agentic AI — 90-Day Journey

A self-directed 90-day project to design, build, and ship AI agents from first principles — starting with no agentic AI code and ending with a deployed working agent.

**Approach:** 15–30 minutes daily. Concepts first, then code, then frameworks, then production.

---

## Status

| Phase                         | Days  | Focus                                                  | Status         |
| ----------------------------- | ----- | ------------------------------------------------------ | -------------- |
| **1 — Foundations**           | 1–21  | Concepts, LLM internals, prompting, ReAct, memory, RAG | ✅ Complete    |
| **2 — Hands-on Basics**       | 22–45 | Building agents from scratch in Python                 | ✅ Complete    |
| **3 — Frameworks & Patterns** | 46–66 | LangGraph, CrewAI, advanced RAG                        | 🚧 In progress |
| **4 — Production & Project**  | 67–90 | Evaluation, safety, deployment, shipping               | ⏳ Upcoming    |

---

## Repository Structure

```
agentic-ai-90day/
├── phase2_basics/        # Days 22–45 — building agents from scratch in Python
├── phase3_frameworks/    # Days 46–66 — LangGraph, CrewAI, RAG (in progress)
├── phase4_project/       # Days 67–90 — final shipped project (upcoming)
├── docs/                 # Phase reference documents & learnings
└── .env.example          # API key template (copy to .env, then fill in)
```

Each `phaseN_.../dayXX_topic/` folder contains the code from that day, following a natural progression: single API call → conversation loop → tool use → multi-tool → real APIs → ReAct loop → error handling → structured logging → refactor → context management → vector memory.

---

## Tech Stack

- **Language:** Python 3.12
- **LLMs:** Gemini (`gemini-3.1-flash-lite` free tier during Phase 2)
- **Vector DB:** Chroma (local, embedded)
- **Frameworks:** LangGraph, CrewAI (Phase 3)
- **Environment management:** `venv` + `direnv` for secrets
- **Deployment:** Hugging Face Spaces or Replit (Phase 4)

---

## Running the Code Locally

```bash
git clone https://github.com/anoop-dryad/agentic-ai-90day.git
cd agentic-ai-90day

# Set up a virtual environment
python3 -m venv agents-env
source agents-env/bin/activate           # macOS/Linux
# agents-env\Scripts\activate            # Windows

# Configure your API key
cp .envrc.example .envrc
# edit .envrc with your real key, then:
direnv allow

# Install dependencies
pip install -r requirements.txt

# Run any day's example
cd phase2_basics/day41_refactor
python agent.py
```

Every day's folder is self-contained and runnable. Later days build on earlier ones by copying + extending, so you can pick any folder and study that day's specific concept.

---

## Design Principles

Documented as they emerged, not decided upfront:

- **Errors are data, not exceptions.** Tools return `{"error": ...}` so the LLM can adapt; raised exceptions kill the loop.
- **Small files, one reason to change.** Tools, config, logging, and the loop each live in their own module.
- **The system prompt is the tool's authority.** Without an explicit _"you MUST call the tool,"_ the LLM answers from training data and bypasses your architecture.
- **Every loop has a max iteration cap.** Removing it is never worth the risk.
- **Log to file, not just stdout.** `print()` dies with the terminal.

---

## Phase Documentation

- **Phase 1 (Days 1–21):** [detailed reference document](docs/Phase_1_Reference_Detailed.docx) — every concept explained
- **Phase 2 (Days 22–45):** [phase README](phase2_basics/README.md) — code walkthrough
- **Lessons across the whole journey:** [LEARNINGS.md](LEARNINGS.md)

---

## License

MIT

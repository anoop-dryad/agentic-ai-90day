# 🤖 Agentic AI — 90 Day Journey

A self-directed 90-day journey to learn how to design, build, and ship AI agents — from the conceptual foundations through to a deployed, working agent.

**Daily commitment:** 15–30 minutes
**Started:** June 2026

---

## 📅 Progress

- ✅ **Phase 1 — Foundations (Days 1–21):** Concepts, LLM internals, prompting, ReAct, memory, RAG
- 📍 **Phase 2 — Hands-On Basics (Days 22–45):** Python implementation of every concept from Phase 1
- ⏳ **Phase 3 — Frameworks & Patterns (Days 46–66):** LangGraph, CrewAI, advanced RAG
- ⏳ **Phase 4 — Production & My Project (Days 67–90):** Evaluation, safety, deployment, shipping a real agent

---

## 📁 Repository Structure

Each `dayXX_topic/` folder contains the code I wrote that day and a short `notes.md` capturing what I learned.

---

## 🛠️ Tech Stack (evolving)

- **Language:** Python 3.12
- **LLMs:** Claude (Anthropic), with experiments on GPT and Gemini
- **Vector DB:** Chroma (local)
- **Frameworks:** LangGraph, CrewAI (Phase 3)
- **Deployment:** Hugging Face Spaces or Replit (Phase 4)

---

## 🚀 Running the code locally

```bash
git clone https://github.com/YOUR_USERNAME/agentic-ai-90day.git
cd agentic-ai-90day

# Set up a virtual environment
python -m venv agents-env
source agents-env/bin/activate     # Mac/Linux
# OR: agents-env\Scripts\activate  # Windows

# Configure your API key
cp .env.example .env
# Then edit .env with your real key

# Install dependencies (from the relevant phase)
pip install -r phase2_basics/requirements.txt
```

---

## 🎯 The Goal

By Day 90, ship a real, deployed AI agent that I built end-to-end and can demo publicly.


from crewai import LLM
import re
from crewai.flow.flow import Flow, start, router, listen, or_
import os
from pydantic import BaseModel
from tools import lookup_invoice, lookup_error_code

llm = LLM(
    model='gemini/gemini-3.1-flash-lite', 
    temperature=0, 
    api_key=os.getenv("GEMINI_API_KEY")
)

class SupportState(BaseModel):
    question: str = ""
    category: str = ""
    backend_data: str = ""
    backend_ok: bool = False
    answer: str = ""

class SupportFlow(Flow[SupportState]):

    @start()
    def classify(self):
        prompt = (
            "Classify this support question. Reply with EXACTLY one word: "
            "billing, technical, or other. No other text.\n\n"
            f"Question: {self.state.question}\n\nCategory:"
        )
        resp = llm.call(prompt)
        print(f"[classify] RAW: {repr(resp)}")   # keep this until it works

        text = str(resp).strip().lower()
        if "billing" in text:
            cat = "billing"
        elif "tech" in text:
            cat = "technical"
        else:
            cat = "other"

        self.state.category = cat
        print(f"[classify] → {cat}")

    @router(classify)
    def route_by_category(self):
        print(f"[router] on category={self.state.category}")
        return self.state.category

    @listen("billing")
    def check_billing(self):
        m = re.search(r"INV-\d+", self.state.question, re.IGNORECASE)
        inv_id = m.group(0) if m else ""
        if not inv_id:
            self.state.backend_ok = False
            self.state.backend_data = "no invoice id found"
            print("[check_billing] no ID → gate FAILS")
            return "checked"
        result = lookup_invoice.run(inv_id)
        self.state.backend_ok = not result.startswith("ERROR")
        self.state.backend_data = result
        print(f"[check_billing] gate {'PASSES' if self.state.backend_ok else 'FAILS'}: {result}")
        return "checked"

    @listen("technical")
    def check_technical(self):
        words = self.state.question.replace("?", " ").split()
        code = next((w for w in words if w.upper().startswith("E") and w[1:].isdigit()), "")
        if not code:
            self.state.backend_ok = False
            self.state.backend_data = "no error code found"
            print("[check_technical] no code → gate FAILS")
            return "checked"
        result = lookup_error_code.run(code)
        self.state.backend_ok = not result.startswith("ERROR")
        self.state.backend_data = result
        print(f"[check_technical] gate {'PASSES' if self.state.backend_ok else 'FAILS'}")
        return "checked"

    @listen("other")
    def handle_other(self):
        self.state.backend_ok = False
        self.state.backend_data = "no backend lookup for this category"
        print("[handle_other] no backend path")
        return "checked"

    @listen(or_(check_billing, check_technical, handle_other))
    def compose_answer(self):
        if self.state.backend_ok:
            prompt = (
                f"Customer asked: {self.state.question}\n"
                f"Verified data: {self.state.backend_data}\n\n"
                "Reply warmly using ONLY the verified data. Add no facts not present."
            )
        else:
            prompt = (
                f"Customer asked: {self.state.question}\n"
                f"Could not verify: {self.state.backend_data}\n\n"
                "Reply warmly, honestly say we couldn't confirm the details and are "
                "escalating to a human. Do NOT guess or reassure about actual status."
            )
        self.state.answer = str(llm.call(prompt)).strip()
        print(f"[compose_answer] done")
        return self.state.answer

if __name__ == "__main__":
    tests = [
        "What's the status of invoice INV-002?",   # billing, clean → gate passes
        "What about invoice INV-999?",             # billing, bad ID → gate fails
        "What does error E429 mean?",              # technical, clean → gate passes
        "My dashboard looks weird.",               # other → no backend path
    ]

    for i, q in enumerate(tests, 1):
        print(f"\n{'═' * 64}\nTEST {i}: {q}\n{'─' * 64}")
        flow = SupportFlow()
        flow.kickoff(inputs={"question": q})
        print(f"\n🤖 {flow.state.answer}\n")



# --------------- workflow--------------------------------
#
# kickoff(question)
#    │  writes question → state
#    ▼
# classify              [LLM call #1: classify]
#    │  writes category → state
#    ▼
# route_by_category     [@router: returns category as an EVENT]
#    │  emits event "technical"
#    ▼
# check_technical       [@listen("technical"): TOOL call + GATE]
#    │  writes backend_ok + backend_data → state
#    │  returns "checked"
#    ▼
# compose_answer        [@listen(or_...): LLM call #2, reads the gate]
#    │  writes answer → state
#    ▼
# flow.state.answer  →  printed to user
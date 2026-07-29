import os
from crewai import Agent, Task, Crew, Process, LLM

from tools import (
    lookup_invoice, get_account_balance,
    lookup_error_code, check_service_status,
)

llm = LLM(
    model="gemini/gemini-3.1-flash-lite",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,
)

billing_agent = Agent(
    role="Billing Specialist",
    goal="Answer billing questions using real invoice and account data only",
    backstory=(
        "You handle invoices, payments, and account balances. You never guess "
        "amounts or statuses — you always look them up. If asked about technical "
        "errors or service status, you say it's outside your scope and answer "
        "only the billing portion."
    ),
    tools=[lookup_invoice, get_account_balance],
    llm=llm,
    allow_delegation=False,
    verbose=True,
)

tech_agent = Agent(
    role="Technical Support Specialist",
    goal="Answer technical questions using real error and status data only",
    backstory=(
        "You handle error codes, bugs, and service status. You never guess what "
        "an error means — you always look it up. If asked about billing or "
        "payments, you say it's outside your scope and answer only the "
        "technical portion."
    ),
    tools=[lookup_error_code, check_service_status],
    llm=llm,
    allow_delegation=False,
    verbose=True,
)

def run_sequential(question: str) -> str:
    """Both specialists work in order, second sees the first's output."""
    billing_task = Task(
        description=(
            f"Customer question: {question}\n\n"
            "Answer ONLY the billing aspects. If there are none, say so briefly."
        ),
        expected_output="A short factual statement about the billing aspects, or 'no billing aspects'.",
        agent=billing_agent,
    )
    tech_task = Task(
        description=(
            f"Customer question: {question}\n\n"
            "Answer ONLY the technical aspects. If there are none, say so briefly."
        ),
        expected_output="A short factual statement about the technical aspects, or 'no technical aspects'.",
        agent=tech_agent,
    )
    synthesis_task = Task(
        description=(
            f"Customer question: {question}\n\n"
            "Using the specialists' findings above, write one clear customer-facing "
            "answer. Do not mention internal teams. Do not invent connections between "
            "issues that the findings don't support."
        ),
        expected_output="A warm, concise reply under 120 words.",
        agent=billing_agent,
        context=[billing_task, tech_task],
    )

    crew = Crew(
        agents=[billing_agent, tech_agent],
        tasks=[billing_task, tech_task, synthesis_task],
        process=Process.sequential,
        verbose=True,
    )
    return str(crew.kickoff())

def run_hierarchical(question: str) -> str:
    """A manager agent decides who handles what."""
    task = Task(
        description=(
            f"Customer question: {question}\n\n"
            "Consult whichever specialists are needed. Do not consult a specialist "
            "whose domain the question doesn't touch. Then produce one clear "
            "customer-facing answer. Do not invent connections between issues."
        ),
        expected_output="A warm, concise reply under 120 words, grounded only in specialist findings.",
    )

    crew = Crew(
        agents=[billing_agent, tech_agent],
        tasks=[task],
        process=Process.hierarchical,
        manager_llm=llm,
        verbose=True,
    )
    return str(crew.kickoff())

if __name__ == "__main__":
    import time

    tests = [
        "What's the status of invoice INV-002?",
        "I'm getting error E429 during checkout AND my invoice INV-002 shows "
        "unpaid. Are these related?",
        "Is the API up, and how much do I owe on CUST-42?",
    ]

    for i, q in enumerate(tests, 1):
        print(f"\n{'═' * 70}\nTEST {i}: {q}\n{'─' * 70}")
        print(f"\n🤖 {run_hierarchical(q)}\n")
        if i < len(tests):
            time.sleep(5)
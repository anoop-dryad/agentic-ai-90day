from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

import os
model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,
)

# A tool — plain Python function, no schema needed
def get_current_time(city: str) -> str:
    """Return current time in a given city.

    Args:
        city: The city to look up, e.g. "Berlin".
    """
    fake_times = {
        "berlin": "15:42 CEST",
        "tokyo": "22:42 JST",
        "london": "14:42 BST",
    }
    return fake_times.get(city.lower(), f"Sorry, I don't know the time in {city}")

agent = create_agent(
    model=model,
    tools=[get_current_time],
    system_prompt="You are a helpful assistant. Use tools when asked about time in a city.",
)

if __name__ == "__main__":
    tests = [
        "What time is it in Berlin?",
        "What time is it in Berlin AND Tokyo?",
        "What's the capital of France?",
    ]
    for q in tests:
        print(f"\n🧑 {q}")
        result = agent.invoke(
            {"messages": [{"role": "user", "content": q}]}
        )
        print(f"🤖 {result['messages'][-1].content}")
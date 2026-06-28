import os
from google import genai
from google.genai import types
from typing import List

MODEL = "gemini-3.1-flash-lite"
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
SYSTEM_PROMPT = """
You are a helpful, concise assistant. Keep replies under 100 words.
""".strip()

memory: list = []

def print_memory_snapshot(history: list, turn: int) -> None:
    """Print what the LLM is about to see this turn."""
    print(f"\n ----- 🧠 MEMORY SNAPSHOT (turn {turn}) ------ ")
    print(f" Total messages in history: {len(memory)}")

    # rough token count - actual token count differs, but this gives the shape
    total_chars = sum(
        len(part.text or "") 
        for content in memory
        for part in (content.parts or []))
    
    approx_tokens = total_chars // 4 # 4 characters per token for english
    print(f" Approx total chars: {total_chars} ->  ~{approx_tokens} tokens")
    print(" Messages (role : first 60 chars): ")

    for i, content in enumerate(history):
        first_part_text = content.parts[0].text if content.parts else ""
        preview = (first_part_text or ""[:60].replace("\n", " "))
        print(f"      [{i}] {content.role:5s} : {preview}{'...' if len(first_part_text) > 60 else ''}")
    print("─" * 50)


print("💬 Chat started. Type 'quit' to exit.\n")
turn = 0

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in {"quit", "exit", "bye"}:
        print("👋 Keep coding!")
        break
    if not user_input:
        continue

    turn += 1

    memory.append(
        types.Content(
            role="user",
            parts=[types.Part(text=user_input)]
        )
    )

    print_memory_snapshot(history=memory, turn=turn)

    response = client.models.generate_content(
        model=MODEL, 
        contents=memory, 
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT, 
            temperature=0.5,
        )
    )

    reply = response.text
    print(f"\nTutor: {reply}\n")

    memory.append(
        types.Content(
            role="model",
            parts=[types.Part(text=reply)],
        ),
    )

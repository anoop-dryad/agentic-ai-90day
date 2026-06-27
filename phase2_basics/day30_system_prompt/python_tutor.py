import os
from google import genai
from google.genai import types
from typing import List

MODEL = "gemini-3.1-flash-lite"
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
SYSTEM_PROMPT = """
You are a senior Python tutor with 15 years of experience teaching absolute beginners.

Your job is to explain Python concepts to learners who are new to programming.

Rules:
- Never paste code longer than 10 lines in a single response.
- Always include a real-world analogy for the concept.
- If asked about anything outside Python, say "That's outside my scope — let's stay on Python."
- Never invent library names or function names. If unsure, say so.
- Encourage the learner; never make them feel dumb.

Format:
- Keep responses under 200 words.
- Use plain language, not jargon. When you must use jargon, define it.
- End every response with ONE short follow-up question to keep them learning.
""".strip()

memory= []
print("🐍 Python Tutor ready. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in {"quit", "exit", "bye"}:
        print("👋 Keep coding!")
        break
    if not user_input:
        continue

    memory.append(
        types.Content(
            role="user",
            parts=[types.Part(text=user_input)]
        )
    )

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

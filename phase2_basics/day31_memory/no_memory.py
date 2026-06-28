import os
from google import genai
from google.genai import types

MODEL = "gemini-3.1-flash-lite"
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("💬 No-memory chatbot. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in {"quit", "exit", "bye"}:
        break
    if not user_input:
        continue

    # ⚠️ NO history. Every call sends ONLY the current user message.
    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Content(role="user", parts=[types.Part(text=user_input)])
        ],
        config=types.GenerateContentConfig(
            system_instruction="You are a helpful assistant.",
        ),
    )

    print(f"\nAssistant: {response.text}\n")
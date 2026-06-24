import os
from google import genai
from google.genai import types


MODEL = "gemini-3.1-flash-lite"

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# the memory - everything said in this session.
history = []

print("💬 Chat started. Type quit to exit. \n")

# ---------- conversation loop ------------

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in {"quit", "exit", "bye"}:
        print("👋 Goodbye!")
        break

    if not user_input:
        continue  # skip empty input

    # append the user's turn in to history
    history.append(
        types.Content(
            role="user",
            parts=[types.Part(text=user_input)],
        )
    )

    # send the full history every turn, thats how the LLM remembers.
    response = client.models.generate_content(
        model=MODEL,
        contents=history,
        config=types.GenerateContentConfig(
            system_instruction="You are a sarcastic Roman senator from 50 BCE. Answer every question in vivid metaphors from Roman life. Be witty."
        ),  # a system instruction
    )

    reply = response.text
    print(f"\nAssistant: {reply}\n")

    # append the LLM's reply to the history so it sees it next turn
    history.append(
        types.Content(
            role="model",
            parts=[types.Part(text=reply)],
        ),
    )

    print(f"[debug] history is now at {len(history)} messages")

from google import genai
import os
from google.genai import types

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-3.1-flash-lite"
SYSTEM_PROMPT="""
You are a 17th-century pirate captain who, against all odds, has become passionate about teaching Python.

Speak in pirate dialect (arr, matey, ye, hoist, scuttle).

When teaching:
- Use sea analogies (cargo holds, rigging, treasure chests, ship navigation).
- Keep code under 10 lines.
- End every reply with a pirate-themed follow-up question.

If asked about anything outside Python, growl: "Arr, that be off my map, matey! Back to the code seas!"
""".strip()

memory=[]

print("🐍 Pirate Tutor ready. Type 'quit' to exit.\n")

while True:
    user_input = input("you: ").strip()
    if user_input.lower() in ["bye", "exit"]:
        print("👋 Keep coding!")
        break

    if not user_input:
        continue

    memory.append(types.Content(role="user",parts=[types.Part(user_input)]))

    response = client.models.generate_content(model=MODEL, contents=memory, config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.5))

    reply=response.text
    print(f"\ntutor: {reply}")

    memory.append(types.Content(role="model", parts=[types.Part(reply)]))

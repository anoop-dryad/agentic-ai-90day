from google import genai
import os

MODEL = "gemini-3.1-flash-lite"

# 1. Authenticate using the env var (loaded by direnv)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# send a message
response = client.models.generate_content(
    model=MODEL,
    contents="In exactly 2 sentence, explain what an AI agent is.",
)

# print the replay
print(response.text)

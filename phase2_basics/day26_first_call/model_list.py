from google import genai
import os

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

for models in client.models.list():
    if "generateContent" in models.supported_actions:
        print(models.name)

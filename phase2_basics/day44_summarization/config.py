"""Agent config: model, system prompt, tool wiring."""

from google.genai import types
from tools import DECLARATIONS

MODEL = "gemini-3.1-flash-lite"
MAX_ITERATIONS = 8

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "For any question about the current time in a city, you MUST call "
    "get_current_time — even if the city sounds unusual. "
    "For any coin flip request, you MUST call flip_a_coin. "
    "Do not answer from your own knowledge when a tool applies. "
    "If a tool returns an error, acknowledge it and offer alternatives."
)

GENERATION_CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    tools=[types.Tool(function_declarations=DECLARATIONS)],
    temperature=0,
)
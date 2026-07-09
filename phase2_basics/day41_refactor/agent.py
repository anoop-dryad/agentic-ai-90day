"""The main agent loop — small enough to read at a glance."""

import os
import time
from google import genai
from google.genai import types

from config import MODEL, MAX_ITERATIONS, GENERATION_CONFIG
from tools import FUNCTIONS
from logging_setup import setup_logger

log = setup_logger()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def call_llm(history: list):
    """Retry with exponential backoff on transient failures."""
    for attempt in range(3):
        try:
            return client.models.generate_content(
                model=MODEL, contents=history, config=GENERATION_CONFIG
            )
        except Exception as e:
            wait = 2 ** attempt
            log.warning(f"LLM error ({type(e).__name__}), retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError("LLM failed 3× in a row")


def run_tool(fc) -> dict:
    """Look up the tool, run it, convert crashes to structured errors."""
    if fc.name not in FUNCTIONS:
        log.error(f"Unknown tool: {fc.name}")
        return {"error": f"No tool named '{fc.name}'"}
    try:
        return FUNCTIONS[fc.name](**fc.args)
    except TypeError as e:
        log.error(f"Bad args to {fc.name}: {e}")
        return {"error": f"Bad arguments: {e}"}
    except Exception as e:
        log.error(f"{fc.name} crashed: {type(e).__name__}: {e}")
        return {"error": f"{fc.name} crashed: {e}"}


def ask(user_message: str) -> str:
    history: list = [
        types.Content(role="user", parts=[types.Part(text=user_message)])
    ]

    for iteration in range(MAX_ITERATIONS):
        resp = call_llm(history)
        part = resp.candidates[0].content.parts[0] # type: ignore

        if not part.function_call:
            return part.text or ""

        fc = part.function_call
        log.info(f"iter {iteration + 1}: {fc.name}({dict(fc.args)})") # type: ignore
        result = run_tool(fc)
        log.debug(f"tool result: {result}")

        history.append(resp.candidates[0].content) # type: ignore
        history.append(types.Content(
            role="user",
            parts=[types.Part.from_function_response(name=fc.name, response=result)], # type: ignore
        ))

    return f"⚠️  Hit max iterations ({MAX_ITERATIONS}) without finishing."


if __name__ == "__main__":
    tests = [
        "What time is it in Berlin?",
        "test failure",
        "What time is it in Berlin AND in Tokyo?",
    ]
    for q in tests:
        print(f"\n🧑 {q}")
        print(f"🤖 {ask(q)}")
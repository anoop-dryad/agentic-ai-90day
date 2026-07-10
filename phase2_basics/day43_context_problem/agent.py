"""The main agent loop — small enough to read at a glance."""

import time
import os
from google import genai
from google.genai import types
from google.genai import errors as genair_errors

from config import MODEL, MAX_ITERATIONS, GENERATION_CONFIG
from tools import FUNCTIONS
from logging_setup import setup_logger


def approx_tokens(history: list) -> int:
    """Rough estimate: ~4 chars per token in English. Good enough for a live gauge."""
    total_chars = 0
    for content in history:
        for part in content.parts or []:
            if part.text:
                total_chars += len(part.text)
            # function_calls and function_responses are dicts — approximate their JSON length
            if hasattr(part, "function_call") and part.function_call:
                total_chars += len(str(dict(part.function_call.args or {}))) + len(
                    part.function_call.name
                )
            if hasattr(part, "function_response") and part.function_response:
                total_chars += len(str(part.function_response.response or {}))
    return total_chars // 4


def real_tokens(history: list) -> int:
    """Actual token count from Gemini. One API call, cost ≈ nothing."""
    result = client.models.count_tokens(model=MODEL, contents=history)
    return result.total_tokens


log = setup_logger()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def call_llm(history: list):
    """Retry with exponential backoff on transient failures."""
    for attempt in range(3):
        try:
            return client.models.generate_content(
                model=MODEL, contents=history, config=GENERATION_CONFIG
            )
        except genair_errors.ServerError as e:
            # 5xx — transient, worth retrying
            wait = 2**attempt
            log.warning(f"LLM 5xx ({e}), retry in {wait}s")
            time.sleep(wait)
        except genair_errors.ClientError as e:
            # 4xx — retrying won't help
            log.error(f"LLM 4xx (non-retryable): {e}")
            raise
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
    history: list = [types.Content(role="user", parts=[types.Part(text=user_message)])]

    for iteration in range(MAX_ITERATIONS):
        tk = approx_tokens(history)
        log.info(f"iter {iteration + 1}: history ~{tk} tokens, {len(history)} messages")

        resp = call_llm(history)
        part = resp.candidates[0].content.parts[0]  # type: ignore

        if not part.function_call:
            return part.text or ""

        fc = part.function_call
        log.info(f"iter {iteration + 1}: {fc.name}({dict(fc.args)})")  # type: ignore
        result = run_tool(fc)
        log.debug(f"tool result: {result}")

        history.append(resp.candidates[0].content)  # type: ignore
        history.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(name=fc.name, response=result)
                ],  # type: ignore
            )
        )

    return f"⚠️  Hit max iterations ({MAX_ITERATIONS}) without finishing."


def ask_with_history(history: list) -> str:
    """Same loop as ask(), but continues an existing conversation."""
    for iteration in range(MAX_ITERATIONS):
        tk = approx_tokens(history)
        log.info(f"iter {iteration + 1}: history ~{tk} tokens, {len(history)} messages")

        resp = call_llm(history)
        part = resp.candidates[0].content.parts[0]

        if not part.function_call:
            return part.text or ""

        fc = part.function_call
        log.info(f"  → {fc.name}({dict(fc.args)})")
        result = run_tool(fc)
        log.debug(f"  ← {result}")

        history.append(resp.candidates[0].content)
        history.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(name=fc.name, response=result)
                ],
            )
        )

    return f"⚠️  Hit max iterations ({MAX_ITERATIONS})"


if __name__ == "__main__":
    # Simulate a growing conversation, one question after another,
    # sharing the same history — this is what a real chat session looks like.
    conversation_history: list = []

    questions = [
        "What time is it in Berlin?",
        "And in Tokyo?",
        "And in London?",
        "And in New York?",
        "And in Sydney?",
        "Flip a coin for me.",
        "What was the time in Berlin again?",  # tests memory
        "Flip another coin.",
        "What time is it in Mumbai?",
        "And in Cochin?",
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n─── turn {i} ────────────────────────────────")
        print(f"🧑 {q}")
        conversation_history.append(
            types.Content(role="user", parts=[types.Part(text=q)])
        )
        answer = ask_with_history(conversation_history)
        conversation_history.append(
            types.Content(role="model", parts=[types.Part(text=answer)])
        )
        print(f"🤖 {answer}")
        print(f"   📏 conversation now ~{approx_tokens(conversation_history)} tokens")

        if i < len(questions):
            time.sleep(10)  # ~20 turns/min, safe under 30 req/min cap

    print(f"\n=== FINAL: {real_tokens(conversation_history)} real tokens ===")

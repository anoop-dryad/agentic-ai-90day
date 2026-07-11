"""The main agent loop — small enough to read at a glance."""

import time
import os
from google import genai
from google.genai import types
from google.genai import errors as genair_errors

from config import MODEL, MAX_ITERATIONS, GENERATION_CONFIG
from tools import FUNCTIONS
from logging_setup import setup_logger
from memory import remember, recall, clear

SUMMARY_TRIGGER = 12  # summarize when history has more than this many messages
KEEP_RECENT = 6  # keep last N messages verbatim
TOP_K = 3   # how many past exchanges to inject


def summarize_old_history(old_messages: list) -> str:
    """Ask the LLM to compress old conversation turns into a short paragraph."""
    # Convert the old messages into readable text
    lines = []
    for msg in old_messages:
        role = msg.role
        for part in msg.parts or []:
            if part.text:
                lines.append(f"{role}: {part.text}")
            if hasattr(part, "function_call") and part.function_call:
                lines.append(
                    f"{role}: [called {part.function_call.name}({dict(part.function_call.args or {})})]"
                )
            if hasattr(part, "function_response") and part.function_response:
                lines.append(
                    f"{role}: [tool result: {part.function_response.response}]"
                )

    convo_text = "\n".join(lines)

    prompt = (
        "Summarize the following conversation between a user and an assistant. "
        "Keep the summary factual and concise (under 100 words). "
        "Preserve: user preferences, key facts stated, decisions made, and open questions. "
        "Drop: pleasantries, exact tool arguments, exact tool outputs unless they matter.\n\n"
        f"Conversation:\n{convo_text}\n\nSummary:"
    )

    resp = client.models.generate_content(
        model=MODEL,
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
        config=types.GenerateContentConfig(temperature=0),  # deterministic-ish
    )
    return resp.candidates[0].content.parts[0].text.strip() # type: ignore


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
    return result.total_tokens # type: ignore


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
                    types.Part.from_function_response(name=fc.name, response=result) # type: ignore
                ],  # type: ignore
            )
        )

    return f"⚠️  Hit max iterations ({MAX_ITERATIONS}) without finishing."


def ask_with_history(history: list, user_message: str) -> str:
    """Retrieve relevant past exchanges, then run the ReAct loop."""

    # 🆕 fetch relevant memories before the LLM sees the user message
    past = recall(user_message, top_k=TOP_K)
    if past:
        memory_note = (
            "[Relevant past exchanges from earlier conversations]:\n\n"
            + "\n---\n".join(past)
            + "\n\n[End of past exchanges. Use them only if they're relevant.]"
        )
        # Prepend the memory context as its own user message
        history.insert(0, types.Content(
            role="user", parts=[types.Part(text=memory_note)]
        ))
        log.info(f"🔍 recalled {len(past)} past exchange(s)")

    # Append the actual current user message
    history.append(types.Content(role="user", parts=[types.Part(text=user_message)]))

    # ...existing loop unchanged...
    for iteration in range(MAX_ITERATIONS):
        compressed = maybe_compress(history)
        if compressed is not history:
            history.clear()
            history.extend(compressed)

        tk = approx_tokens(history)
        log.info(f"iter {iteration + 1}: history ~{tk} tokens, {len(history)} messages")

        resp = call_llm(history)
        part = resp.candidates[0].content.parts[0] # type: ignore

        if not part.function_call:
            return part.text or ""

        fc = part.function_call
        log.info(f"  → {fc.name}({dict(fc.args)})") # type: ignore
        result = run_tool(fc)
        log.debug(f"  ← {result}")

        history.append(resp.candidates[0].content) # type: ignore
        history.append(types.Content(
            role="user",
            parts=[types.Part.from_function_response(name=fc.name, response=result)], # type: ignore
        ))

    return f"⚠️  Hit max iterations ({MAX_ITERATIONS})"


def maybe_compress(history: list) -> list:
    """If history is too long, replace old messages with a summary."""
    if len(history) <= SUMMARY_TRIGGER:
        return history

    # split: everything except last KEEP_RECENT gets summarized
    to_summarize = history[:-KEEP_RECENT]
    to_keep = history[-KEEP_RECENT:]

    log.info(f"📦 compressing {len(to_summarize)} old messages...")
    summary = summarize_old_history(to_summarize)
    log.info(f"📦 summary: {summary[:100]}...")

    # replace old messages with a single "note" message
    summary_note = types.Content(
        role="user",
        parts=[types.Part(text=f"[Earlier conversation summary]:\n{summary}")],
    )

    return [summary_note] + to_keep


if __name__ == "__main__":
    import time as _time

    # start fresh so results are reproducible
    clear()
    conversation_history: list = []

    # First 4 questions establish "facts" the agent should later recall
    facts = [
        "What time is it in Berlin?",
        "My favorite pastime is bouldering.",
        "I have a cat named Mochi.",
        "Flip a coin.",
    ]

    # Then a bunch of unrelated turns to grow context
    filler = [
        "What time is it in Tokyo?",
        "And in London?",
        "Flip another coin.",
        "What time is it in Sydney?",
        "And in New York?",
        "Flip another coin.",
    ]

    # Then the recall test: something that requires memory from turn 2 or 3
    recall_tests = [
        "What did I tell you my favorite pastime was?",
        "What was my cat's name?",
    ]

    for i, q in enumerate(facts + filler + recall_tests, 1):
        print(f"\n─── turn {i} ────────────────────────────────")
        print(f"🧑 {q}")
        answer = ask_with_history(conversation_history, q)
        conversation_history.append(
            types.Content(role="model", parts=[types.Part(text=answer)])
        )
        remember(q, answer)   # 🆕 save to vector memory
        print(f"🤖 {answer}")
        print(f"   📏 conversation now ~{approx_tokens(conversation_history)} tokens, "
              f"{len(conversation_history)} messages")
        if i < len(facts + filler + recall_tests):
            _time.sleep(10)   # respect free-tier rate limit
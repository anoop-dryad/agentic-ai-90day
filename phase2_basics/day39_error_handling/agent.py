from google import genai
import os
from google.genai import types
import time
from time_tool import get_current_time, time_tool_declaration
from coin_tool import flip_coin, flip_coin_declaration
from broken_tool import broken_tool, broken_tool_declaration

MODEL = "gemini-3.1-flash-lite"
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
TOOLS = {
    "get_current_time": get_current_time,
    "flip_a_coin": flip_coin,
    "broken_tool": broken_tool,
}
MAX_ITERATIONS = 5

config = types.GenerateContentConfig(
    system_instruction=(
        "You are a helpful assistant with access to tools. "
        "For any question about the current time in a city, you MUST call "
        "get_current_time. For any coin flip request, you MUST call flip_a_coin. "
        "Do not answer from your own knowledge when a tool applies. "
        "If a tool returns an error, acknowledge it and offer alternatives."
        "If test failure is needed, then u must call broken_tool"
    ),
    temperature=0,
    tools=[
        types.Tool(
            function_declarations=[
                time_tool_declaration,
                flip_coin_declaration,
                broken_tool_declaration,
            ]
        ),
    ],
)


def ask(user_message: str) -> str:
    history: list = [
        types.Content(
            role="user",
            parts=[types.Part(text=user_message)],
        )
    ]

    for iteration in range(MAX_ITERATIONS):
        resp = call_llm(history=history)

        part = resp.candidates[0].content.parts[0]
        if not part.function_call:
            return part.text or ""

        function_call = part.function_call
        print(
            f"   🔁 iter {iteration + 1}: {function_call.name}({dict(function_call.args)})"
        )

        # to prevent the llm hallucination sof calling unknown tools
        if function_call.name not in TOOLS:
            return {
                "error": f"No tool named '{function_call.name}'. Available: {list(TOOLS)}"
            }
        else:
            try:
                result = TOOLS[function_call.name](**function_call.args)
            except TypeError as e:
                result = {"error": f"Bad arguments to {function_call.name}: {e}"}
            except Exception as e:
                result = {
                    "error": f"{function_call.name} crashed: {type(e).__name__}: {e}"
                }

            print(f"   📦 {result}")

        history.append(resp.candidates[0].content)
        history.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name=function_call.name,
                        response=result,
                    )
                ],
            )
        )

    return f"⚠️  Hit max iterations ({MAX_ITERATIONS}) without finishing."


def call_llm(history: list):
    for attempt in range(3):
        try:
            return client.models.generate_content(
                model=MODEL,
                contents=history,
                config=config,
            )
        except Exception as e:
            wait = 2 * attempt
            print(f"   ⚠️  LLM error ({type(e).__name__}), retry in {wait}s")
            time.sleep(wait)

    raise RuntimeError("LLM failed 3× in a row")


if __name__ == "__main__":
    tests = [
        "What time is it in Berlin?",
        "test failure",
        "What time is it in Berlin AND in Tokyo?",
    ]

    for ques in tests:
        print(f"\n🧑 {ques}")
        print(f"🤖 {ask(ques)}")

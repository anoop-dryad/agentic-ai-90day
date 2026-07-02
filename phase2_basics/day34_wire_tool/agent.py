from google import genai
import os
from tool_time import get_current_time, get_current_time_declaration
from google.genai import types


MODEL = "gemini-3.1-flash-lite"
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Registry: name (string from LLM) → real Python function
TOOLS = {"get_current_time":get_current_time}

# Config the LLM sees: system prompt + tools + temp 0 (Day 12)
config = types.GenerateContentConfig(
    system_instruction=(
        "You are a helpful assistant. For ANY question about the current time in a named city, "
        "you MUST call get_current_time — even if the city sounds unusual, fictional, or unfamiliar."
        "Let the tool decide whether the city is known. Do not refuse based on your own knowledge."
    ),
    tools=[types.Tool(function_declarations=[get_current_time_declaration])],
    temperature=0
)

def ask(user_message: str) -> str:
    """One-shot: user → LLM → (maybe tool) → LLM → reply."""
    history: list = [types.Content(
        role="user", 
        parts=[types.Part(text=user_message)])
    ]

    # --- Step 3: LLM decides ---
    resp = client.models.generate_content(
        model=MODEL, 
        contents=history, 
        config=config
    )

    # sample LLM response structure
    # {  
    #   "candidates": [{
    #       "content": {
    #          "role": "model",
    #          "parts": [ 
    #               { "text": "It is 3:42 PM in Berlin." }
    #                 # OR
    #               { "function_call": { "name": "get_current_time", "args": {"city": "Berlin"} } }
    #           ]
    #       }
    #    }]
    # }

    # Was a tool requested?
    part = resp.candidates[0].content.parts[0] # type: ignore
    if not part.function_call:
        return part.text or ""
    
    fc = part.function_call
    print(f"   🛠️  LLM wants to call: {fc.name}({dict(fc.args)})") # type: ignore

    tool_fn = TOOLS[fc.name] # type: ignore
    result = tool_fn(**fc.args) # type: ignore

    print(f"   📦 Tool returned: {result}")

    history.append(resp.candidates[0].content) # type: ignore
    history.append(
        types.Content(
            role="user",
            parts=[types.Part.from_function_response(name=fc.name, response=result)], # type: ignore
        )
    )

    # its an explicit choice to show the result, will be replaced with loop calls later.
    resp2 = client.models.generate_content(
        model=MODEL, contents=history, config=config
    )
    return resp2.candidates[0].content.parts[0].text or "" # type: ignore


if __name__ == "__main__":
    tests = [
        "What time is it in Berlin?",       # happy path
        "What time is it in Atlantis?",     # tool returns error, LLM recovers
        "What's the capital of France?",    # no tool needed
    ]
    for q in tests:
        print(f"\n🧑 {q}")
        print(f"🤖 {ask(q)}")
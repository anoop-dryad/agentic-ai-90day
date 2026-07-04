import os
from google.genai import types
from google import genai

from time_tool import get_current_time, get_current_time_declaration
from coin_tool import flip_a_coin, flip_a_coin_declaration

MODEL = "gemini-3.1-flash-lite"
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
TOOLS = {
        "get_current_time": get_current_time,
        "flip_a_coin": flip_a_coin}

config = types.GenerateContentConfig(
    tools=[
        types.Tool(
            function_declarations=[
                get_current_time_declaration,
                flip_a_coin_declaration,
            ]
        )],
    temperature=0,
    system_instruction=(
        "You are a helpful assistant with access to tools. "
        "For any question about the current time in a city, you MUST call "
        "get_current_time. For any coin flip request, you MUST call flip_a_coin. "
        "Do not answer from your own knowledge when a tool applies. "
        "If a tool returns an error, acknowledge it and offer alternatives."
    ))


def ask(user_message:str) -> str:
    history:list = [types.Content(
        role="user", 
        parts=[types.Part(text=user_message)])]
    
    resp = client.models.generate_content(
        model=MODEL,
        config=config, 
        contents=history
    )

    part = resp.candidates[0].content.parts[0] # type: ignore
    if not part.function_call:
        return part.text or ""
    
    fc = part.function_call
    print(f"   🛠️  {fc.name}({dict(fc.args)})") # type: ignore
    result = TOOLS[fc.name](**fc.args) # type: ignore
    print(f"   📦 {result}")

    history.append(resp.candidates[0].content) # type: ignore
    history.append(types.Content(
        role="user",
        parts=[types.Part.from_function_response(name=fc.name, response=result)])) # type: ignore
    
    resp2 = client.models.generate_content(
        model=MODEL, contents=history, config=config
    )
    return resp2.candidates[0].content.parts[0].text or "" # type: ignore


if __name__ == "__main__":
    tests = [
        "What time is it in New Delhi?",     # → time tool
        "Flip a coin for me.",             # → coin tool
        "Should I go for a walk? Flip a coin.",  # → coin tool (implicit)
        "What's the weather like?",       # → neither, no matching tool
        "I need help deciding between two lunch options.",
        "What time should I flip a coin?",
        "Roll a dice."
    ]
    for q in tests:
        print(f"\n🧑 {q}")
        print(f"🤖 {ask(q)}")
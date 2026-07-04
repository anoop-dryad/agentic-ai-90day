import random
from google.genai import types
def flip_a_coin() -> dict:
    """No inputs, no failure modes, just heads or tails."""
    return {
        "result": random.choice(["heads", "tails"])
    }

flip_a_coin_declaration = types.FunctionDeclaration(
    name="flip_a_coin",
    description=(
        "Flip a fair coin and return the result. "
        "Use this when the user asks to flip a coin, make a coin toss, "
        "or wants a 50/50 random choice. "
        "Do NOT use this for other random tasks (dice, cards, numbers). "
        "Returns: {'result': 'heads' | 'tails'}. Never fails."
    ),
    parameters={
        "type":"object",
        "properties":{}
    }, # type: ignore
)

if __name__ == "__main__":
    for i in range(5):
        print(flip_a_coin())
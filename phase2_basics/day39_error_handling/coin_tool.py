import random
from google.genai import types


def flip_coin():
    return {
        "result": random.choices(["head", "tail"]),
    }


flip_coin_declaration = types.FunctionDeclaration(
    name="flip_a_coin",
    description=(
        "Flip a fair coin and return the result. "
        "Use this when the user asks to flip a coin, make a coin toss, "
        "or wants a 50/50 random choice. "
        "Do NOT use this for other random tasks (dice, cards, numbers). "
        "Returns: {'result': 'heads' | 'tails'}. Never fails."
    ),
    parameters={
        "type": "object",
        "properties": {},
    },
)

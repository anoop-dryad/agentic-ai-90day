"""Tool exports."""

from .time_tool import get_current_time, time_tool_declaration
from .coin_tool import flip_coin, flip_coin_declaration
from .broken_tool import broken_tool, broken_tool_declaration

# Registry: what the LLM sees ↔ what your code runs
FUNCTIONS = {
    "get_current_time": get_current_time,
    "flip_a_coin": flip_coin,
    "broken_tool": broken_tool,
}

DECLARATIONS = [
    time_tool_declaration,
    flip_coin_declaration,
    broken_tool_declaration,
]
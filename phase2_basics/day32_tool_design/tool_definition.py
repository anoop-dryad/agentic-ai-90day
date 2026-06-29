from google.genai import types

import textwrap

# this should be used to strip all whitespaces else will count that for token leading to cost.
description = textwrap.dedent("""
        This tool helps us to get the current time of a given city.
        It should return the current local time of the city in standard format.
        Invoke this when user asks specifically for time.
        Should not be used teh city is not specifically added.
    """).strip()


# tool definition : initial version
get_current_time = types.FunctionDeclaration(
    name="get_current_time",
    # 2 WHAT (doing and return) and 2 WHEN (expctation and restriction)
    description=description,
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "Exact city name eg: Berlin, Munich, Cochin",
            },
        },
        "required": ["city"],
    },
)


# suggested change : updated version
get_current_time = types.FunctionDeclaration(
    name="get_current_time",
    description=(
        "Get the current local time for a given city. "
        "Use this when the user asks specifically about the current time "
        "in a named city. "
        "Do NOT use it for historical times, future times, or time-zone "
        "conversions/math. "
        "If no city is specified, ask the user — do not call this tool. "
        "Returns: {'city': str, 'local_time': str (ISO 8601 with offset, "
        "e.g. '2026-06-25T15:42:00+02:00'), 'timezone': str}."
    ),
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": (
                    "Full city name in English, no country or state. "
                    "Examples: 'Berlin', 'Munich', 'Cochin'."
                ),
            },
        },
        "required": ["city"],
    },
)

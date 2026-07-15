"""Tools ported from Day 45. Same functions, same errors-as-data pattern."""

import random
import requests
from langchain_core.tools import tool

# ---------- get_current_time ----------

CITY_TIMEZONES = {
    "berlin":     "Europe/Berlin",
    "munich":     "Europe/Berlin",
    "london":     "Europe/London",
    "tokyo":      "Asia/Tokyo",
    "cochin":     "Asia/Kolkata",
    "kochi":      "Asia/Kolkata",
    "new delhi":  "Asia/Kolkata",
    "mumbai":     "Asia/Kolkata",
    "new york":   "America/New_York",
    "sydney":     "Australia/Sydney",
}

API_URL = "https://timeapi.io/api/time/current/zone?timeZone={tz}"
TIMEOUT_SECONDS = 5


@tool
def get_current_time(city: str) -> dict:
    """
    Get the current local time for a given city.

    Use this for CURRENT time only — not historical, not forecast,
    not timezone math. If no city is specified, ask the user first.

    Args:
        city: Full city name in English. Examples: 'Berlin', 'Mumbai'.
    """
    if not city or not isinstance(city, str):
        return {"error": "Missing or invalid 'city' argument."}

    key = city.strip().lower()
    tz = CITY_TIMEZONES.get(key)
    if not tz:
        return {
            "error": f"City '{city}' not in my known city list.",
            "known_cities": sorted(CITY_TIMEZONES.keys()),
        }

    try:
        resp = requests.get(API_URL.format(tz=tz), timeout=TIMEOUT_SECONDS)
    except requests.Timeout:
        return {"error": f"Timezone service timed out after {TIMEOUT_SECONDS}s."}
    except requests.RequestException as e:
        return {"error": f"Network error: {e}"}

    if resp.status_code != 200:
        return {"error": f"API returned status {resp.status_code}."}

    try:
        data = resp.json()
    except ValueError:
        return {"error": "API returned invalid JSON."}

    return {
        "city": city.strip().title(),
        "local_time": data["dateTime"],
        "timezone": data["timeZone"],
    }


# ---------- flip_a_coin ----------

@tool
def flip_a_coin() -> dict:
    """
    Flip a fair coin. Use ONLY when the user asks for a coin flip.
    Do NOT use for dice, cards, or other random tasks.

    Returns: {'result': 'heads' | 'tails'}. Never fails.
    """
    return {"result": random.choice(["heads", "tails"])}


# ---------- broken_tool (for error handling test) ----------

@tool
def broken_tool() -> dict:
    """
    A tool that always fails. Use ONLY when the user says 'test failure'.

    This exists to prove the agent recovers when a tool crashes.
    """
    raise ValueError("simulated crash")
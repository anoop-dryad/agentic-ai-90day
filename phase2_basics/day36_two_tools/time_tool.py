from google.genai import types

CITY_TIMES = {
    "berlin":  ("2026-06-25T15:42:00+02:00", "Europe/Berlin"),
    "munich":  ("2026-06-25T15:42:00+02:00", "Europe/Berlin"),
    "cochin":  ("2026-06-25T19:12:00+05:30", "Asia/Kolkata"),
    "kochi":   ("2026-06-25T19:12:00+05:30", "Asia/Kolkata"),
    "tokyo":   ("2026-06-25T22:42:00+09:00", "Asia/Tokyo"),
    "london":  ("2026-06-25T14:42:00+01:00", "Europe/London"),
}

def get_current_time(city:str) -> dict:
    if not city or not isinstance(city, str):
        return {"error": "Missing or invalid city argument"}

    key = city.strip().lower()
    if key not in CITY_TIMES:
        return {
            "error": f"City '{city}' not found in the mock database.",
            "known_cities": sorted(CITY_TIMES.keys()),
        }
    
    local_time, timezone = CITY_TIMES[key]
    return {
        "city": city.strip().title(),
        "local_time": local_time,
        "timezone": timezone,
    }

get_current_time_declaration = types.FunctionDeclaration(
    name="get_current_time",
    description=(
        "Get the current local time for a given city. "
        "Use this when the user asks about the current time in a named city. "
        "Do NOT use it for historical times, future times, or time-zone "
        "conversions/math. If no city is specified, ask the user — do not "
        "call this tool. "
        "Returns: {'city': str, 'local_time': str (ISO 8601 with offset), "
        "'timezone': str}. On failure: {'error': str}."
    ),
    parameters={
        "type":"object",
        "properties":{
            "city":{
                "type":"string",
                "description":(
                    "Full city name in English, no country or state. "
                    "Examples: 'Berlin', 'Munich', 'Cochin'."
                )
            }

        },
        "required":["city"]
        
    } # type: ignore
)
from google.genai import types
import requests

CITY_TIMEZONES = {
    "berlin":     "Europe/Berlin",
    "munich":     "Europe/Berlin",
    "hamburg":    "Europe/Berlin",
    "london":     "Europe/London",
    "tokyo":      "Asia/Tokyo",
    "cochin":     "Asia/Kolkata",
    "kochi":      "Asia/Kolkata",
    "new delhi":  "Asia/Kolkata",
    "mumbai":     "Asia/Kolkata",
    "new york":   "America/New_York",
    "los angeles":"America/Los_Angeles",
    "sydney":     "Australia/Sydney",
}
API_URL = "https://timeapi.io/api/time/current/zone?timeZone={tz}"
TIMEOUT_SECONDS = 5

def get_current_time(city:str) -> dict:
    if not city or not isinstance(city, str):
        return {"error": "Missing or invalid city argument"}

    key = city.strip().lower()
    tz = CITY_TIMEZONES.get(key)
    if not tz:
        return {
            "error": f"City '{city}' not found in the mock database.",
            "known_cities": sorted(CITY_TIMEZONES.keys()),
        }
    
    try:
        resp = requests.get(url=API_URL.format(tz=tz), timeout=TIMEOUT_SECONDS)
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
        "utc_offset": data.get("utcOffset", ""),
    }


get_current_time_declaration = types.FunctionDeclaration(
    name="get_current_time",
    description=(
        "Get the current local time for a given city. "
        "Use this when the user asks about the current time in a named city. "
        "Do NOT use it for historical times, future times, or timezone math. "
        "If no city is specified, ask the user — do not call this tool. "
        "Returns: {'city', 'local_time' (ISO 8601), 'timezone', 'utc_offset'}. "
        "On failure: {'error': str}."
    ),
    parameters={
        "type":"object",
        "properties":{
            "city":{
                "type":"string",
                "description":(
                    "Full city name in English. "
                    "Examples: 'Berlin', 'Munich', 'Cochin'."
                )
            }

        },
        "required":["city"]
        
    } # type: ignore
)

if __name__ == "__main__":
     tests = ["Berlin", "New Delhi", "Atlantis", "", None]
     for test in tests:
         print(f"{test!r:12} → {get_current_time(test)}")  # type: ignore[arg-type]
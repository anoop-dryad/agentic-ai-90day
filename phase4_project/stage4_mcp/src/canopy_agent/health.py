"""Device health judgment. Pure functions over device data — no LLM, no I/O.
Deterministic and exhaustively testable. The LLM never decides health;
it only describes what these functions compute."""

from datetime import datetime, timezone

from canopy_agent.config import settings


def parse_last_seen(last_seen: str) -> datetime | None:
    """Parse an RFC3339 timestamp. Returns None if unparseable."""
    try:
        # handle both '...Z' and '+00:00' forms
        return datetime.fromisoformat(last_seen.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def compute_health(device: dict) -> dict:
    """Given verified device data, compute health flags.

    Returns a dict of booleans + context the LLM will describe.
    Every flag is a deterministic judgment, not the LLM's opinion.
    """
    flags = {
        "is_stale": False,
        "low_battery": False,
        "freshness_unknown": False,
        "status_field": device.get("status", "unknown"),
        "battery_pct": device.get("battery_pct"),
        "minutes_since_seen": None,
        "concerns": [],  # human-readable concern strings, in priority order
    }

    # --- staleness ---
    last_seen = parse_last_seen(device.get("last_seen", ""))
    if last_seen is not None:
        now = datetime.now(timezone.utc)
        minutes = (now - last_seen).total_seconds() / 60
        flags["minutes_since_seen"] = round(minutes, 1)
        if minutes > settings.STALE_THRESHOLD_MINUTES:
            flags["is_stale"] = True
            flags["concerns"].append(
                f"has not reported in {round(minutes)} minutes "
                f"(threshold {settings.STALE_THRESHOLD_MINUTES}m) — "
                f"status may be unreliable"
            )
    else:
        # can't determine freshness — treat as a concern, not as "fine"
        flags["freshness_unknown"] = True
        flags["concerns"].append("last-seen time is missing or invalid")

    # --- battery ---
    battery = device.get("battery_pct")
    if isinstance(battery, (int, float)) and battery <= settings.LOW_BATTERY_PCT:
        flags["low_battery"] = True
        flags["concerns"].append(f"battery is low at {battery}%")

    # --- the key derived judgment ---
    # A device is "healthy" only if status is online AND not stale AND battery ok.
    flags["healthy"] = (
        flags["status_field"] == "online"
        and not flags["is_stale"]
        and not flags["freshness_unknown"]
        and not flags["low_battery"]
    )

    return flags


def test_real_backend_timestamp_format_parses():
    """Backend returns +02:00 offset, not Z. Must parse, not fall to unknown."""
    d = {
        "status": "online",
        "battery_pct": 87,
        "last_seen": "2026-09-01T13:15:36+02:00",
    }
    h = compute_health(d)
    assert h["freshness_unknown"] is False  # parsed correctly
    assert h["minutes_since_seen"] is not None

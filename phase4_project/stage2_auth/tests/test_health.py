"""Health judgment tests. Pure functions — fast, deterministic, exhaustive."""

from datetime import datetime, timedelta, timezone

from canopy_agent.health import compute_health


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _recent(minutes_ago: float) -> str:
    return _iso(datetime.now(timezone.utc) - timedelta(minutes=minutes_ago))


def test_healthy_device():
    d = {"status": "online", "battery_pct": 87, "last_seen": _recent(2)}
    h = compute_health(d)
    assert h["healthy"] is True
    assert h["is_stale"] is False
    assert h["low_battery"] is False
    assert h["concerns"] == []


def test_online_but_stale_is_not_healthy():
    """The Stage 0 gap: online status, but hasn't reported in 3 hours."""
    d = {"status": "online", "battery_pct": 87, "last_seen": _recent(180)}
    h = compute_health(d)
    assert h["is_stale"] is True
    assert h["healthy"] is False  # ← NOT healthy despite 'online'
    assert any("not reported" in c for c in h["concerns"])


def test_online_but_low_battery_is_not_healthy():
    """The dev-002 case: online, but 12% battery."""
    d = {"status": "online", "battery_pct": 12, "last_seen": _recent(2)}
    h = compute_health(d)
    assert h["low_battery"] is True
    assert h["healthy"] is False  # ← NOT healthy despite 'online'
    assert any("battery" in c for c in h["concerns"])


def test_stale_and_low_battery_both_flagged():
    d = {"status": "online", "battery_pct": 8, "last_seen": _recent(200)}
    h = compute_health(d)
    assert h["is_stale"] is True
    assert h["low_battery"] is True
    assert len(h["concerns"]) >= 2  # both concerns present


def test_offline_is_not_healthy():
    d = {"status": "offline", "battery_pct": 90, "last_seen": _recent(1)}
    h = compute_health(d)
    assert h["healthy"] is False


def test_missing_last_seen_is_a_concern_not_fine():
    """Can't determine freshness → treat as concern, never as healthy."""
    d = {"status": "online", "battery_pct": 90, "last_seen": "garbage"}
    h = compute_health(d)
    assert h["minutes_since_seen"] is None
    assert h["healthy"] is False  # unknown freshness ≠ healthy
    assert any("last-seen" in c.lower() for c in h["concerns"])


def test_boundary_just_under_threshold_is_ok():
    """14 min (under 15) → not stale."""
    d = {"status": "online", "battery_pct": 90, "last_seen": _recent(14)}
    h = compute_health(d)
    assert h["is_stale"] is False


def test_boundary_just_over_threshold_is_stale():
    """16 min (over 15) → stale."""
    d = {"status": "online", "battery_pct": 90, "last_seen": _recent(16)}
    h = compute_health(d)
    assert h["is_stale"] is True

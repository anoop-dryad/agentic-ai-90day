"""Tests for the gate logic. Deterministic, no LLM, no real backend.

These are the safety tests. Each one pins a specific failure mode of the
gate: the behaviors that must NEVER regress, because a broken gate means
a fabricated device status reaches a customer.
"""

import requests
import responses
from canopy_agent.backend import get_device
from canopy_agent.config import settings

# ---------- happy path ----------


@responses.activate
def test_valid_device_passes_gate():
    """A clean 200 with all fields → gate passes, data returned."""
    responses.add(
        responses.GET,
        f"{settings.BACKEND_BASE_URL}/{settings.DEVICE_PATH}/dev-001",
        json={
            "id": "dev-001",
            "name": "North Ridge",
            "status": "online",
            "battery_pct": 87,
            "last_seen": "2026-09-01T13:15:00Z",
        },
        status=200,
    )
    result = get_device("dev-001")
    assert result.ok is True
    assert result.data["id"] == "dev-001"
    assert result.data["status"] == "online"


# ---------- the SAFETY tests: every failure must fail the gate ----------


@responses.activate
def test_not_found_fails_gate():
    """404 → gate fails, marked not_found (a real fact, not a verify failure)."""
    responses.add(
        responses.GET,
        f"{settings.BACKEND_BASE_URL}/{settings.DEVICE_PATH}/dev-999",
        json={"error": "device not found", "code": "DEVICE_NOT_FOUND"},
        status=404,
    )
    result = get_device("dev-999")
    assert result.ok is False
    assert result.data == {"not_found": True}


@responses.activate
def test_server_error_fails_gate():
    """500 → gate fails. Backend broke; we cannot trust anything."""
    responses.add(
        responses.GET, f"{settings.BACKEND_BASE_URL}/devices/dev-001", status=500
    )
    result = get_device("dev-001")
    assert result.ok is False
    assert "500" in result.reason


@responses.activate
def test_invalid_json_fails_gate():
    """200 but body isn't JSON → gate fails. 'Succeeded' but garbage."""
    responses.add(
        responses.GET,
        f"{settings.BACKEND_BASE_URL}/{settings.DEVICE_PATH}/dev-001",
        body="this is not json",
        status=200,
    )
    result = get_device("dev-001")
    assert result.ok is False
    assert "invalid json" in result.reason.lower()


@responses.activate
def test_missing_field_fails_gate():
    """200, valid JSON, but missing last_seen → gate fails.
    The insidious case: technically-successful, incomplete data."""
    responses.add(
        responses.GET,
        f"{settings.BACKEND_BASE_URL}/{settings.DEVICE_PATH}/dev-001",
        json={
            "id": "dev-001",
            "name": "North Ridge",
            "status": "online",
            "battery_pct": 87,
        },  # no last_seen
        status=200,
    )
    result = get_device("dev-001")
    assert result.ok is False
    assert "last_seen" in result.reason


@responses.activate
def test_timeout_fails_gate():
    """Backend times out → gate fails, does NOT raise.
    This is the backend-down case that must produce honest escalation."""
    responses.add(
        responses.GET,
        f"{settings.BACKEND_BASE_URL}/{settings.DEVICE_PATH}/dev-001",
        body=requests.exceptions.Timeout(),
    )
    result = get_device("dev-001")
    assert result.ok is False
    assert "timed out" in result.reason.lower()


@responses.activate
def test_connection_error_fails_gate():
    """Backend unreachable → gate fails gracefully, no exception escapes."""
    responses.add(
        responses.GET,
        f"{settings.BACKEND_BASE_URL}/{settings.DEVICE_PATH}/dev-001",
        body=requests.exceptions.ConnectionError(),
    )
    result = get_device("dev-001")
    assert result.ok is False
    assert result.data is None

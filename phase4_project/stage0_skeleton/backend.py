"""Backend client for the canopy device API. The gate lives here."""

import requests

BASE_URL = "http://localhost:8080/v1"
TIMEOUT_SECONDS = 5


class GateResult:
    """The result of a gated backend call.

    ok=True  → data is trustworthy, LLM may characterize it
    ok=False → could NOT verify; LLM must NOT invent, agent escalates
    """

    def __init__(self, ok: bool, data: dict | None = None, reason: str = ""):
        self.ok = ok
        self.data = data
        self.reason = reason


def get_device(device_id: str) -> GateResult:
    """Fetch a device and GATE the response.

    Returns GateResult, never raises — failures become ok=False,
    so the agent can honestly escalate instead of fabricating.
    """
    url = f"{BASE_URL}/devices/{device_id}"

    try:
        resp = requests.get(url, timeout=TIMEOUT_SECONDS)
    except requests.Timeout:
        return GateResult(False, reason=f"backend timed out after {TIMEOUT_SECONDS}s")
    except requests.RequestException as e:
        return GateResult(False, reason=f"backend unreachable: {e}")

    # GATE 1: not found — a REAL answer ("this device doesn't exist"),
    # distinct from "couldn't check". 404 with your structured error body.
    if resp.status_code == 404:
        return GateResult(
            False, reason=f"device '{device_id}' not found", data={"not_found": True}
        )

    # GATE 2: any non-200 → could not verify
    if resp.status_code != 200:
        return GateResult(False, reason=f"backend returned status {resp.status_code}")

    # GATE 3: body must parse and contain the fields we rely on
    try:
        body = resp.json()
    except ValueError:
        return GateResult(False, reason="backend returned invalid JSON")

    required = {"id", "name", "status", "battery_pct", "last_seen"}
    missing = required - body.keys()
    if missing:
        return GateResult(False, reason=f"backend response missing fields: {missing}")

    # Passed every gate — data is trustworthy.
    return GateResult(True, data=body)


def list_devices() -> GateResult:
    """List all devices, gated the same way."""
    url = f"{BASE_URL}/devices"
    try:
        resp = requests.get(url, timeout=TIMEOUT_SECONDS)
    except requests.RequestException as e:
        return GateResult(False, reason=f"backend unreachable: {e}")

    if resp.status_code != 200:
        return GateResult(False, reason=f"backend returned status {resp.status_code}")
    try:
        body = resp.json()
    except ValueError:
        return GateResult(False, reason="backend returned invalid JSON")

    return GateResult(True, data=body)

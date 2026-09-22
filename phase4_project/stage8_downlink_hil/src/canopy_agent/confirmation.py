"""Confirmation tokens for gated writes (HIL enforcement).

A token proves a proposal happened for a specific device+command, recently.
send_downlink requires a valid token, so the confirmation gate is structural,
not a client convention. Stateless (HMAC-signed) — no server memory needed."""

import hashlib
import hmac
import time

from canopy_agent.config import settings

_TTL_SECONDS = 120


def make_confirmations_token(device_id: str, command: str) -> str:
    """Mint a signed, action-bound, time-limited token for a proposed downlink."""
    issued_at = str(int(time.time()))
    payload = f"{device_id}:{command}:{issued_at}"
    signature = hmac.new(
        settings.CONFIRMATION_SECRET.encode(), payload.encode(), hashlib.sha256
    ).hexdigest()

    return f"{payload}:{signature}"


def valid_confirmation_token(token: str, device_id: str, command: str) -> bool:
    """Verify: right device+command, not expired, valid signature."""

    try:
        dev, cmd, issued_at, signature = token.split(":", 3)
    except ValueError:
        return False

    if dev != device_id or cmd != command:
        return False

    if int(time.time()) - int(issued_at) > _TTL_SECONDS:
        return False

    payload = f"{dev}:{cmd}:{issued_at}"
    expected = hmac.new(
        settings.CONFIRMATION_SECRET.encode(),
        payload.encode(),
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(signature, expected)

import time

from canopy_agent.confirmation import (
    make_confirmations_token,
    valid_confirmation_token,
)


def test_valid_token_passes():
    t = make_confirmations_token("dev-003", "reset")
    assert valid_confirmation_token(t, "dev-003", "reset") is True


def test_wrong_command_fails():
    t = make_confirmations_token("dev-003", "reset")
    assert valid_confirmation_token(t, "dev-003", "calibrate") is False  # action-bound


def test_forged_token_fails():
    assert (
        valid_confirmation_token(
            "dev-003:reset:123:FAKE",
            "dev-003",
            "reset",
        )
        is False
    )


def test_expired_token_fails(monkeypatch):
    import canopy_agent.confirmation as c

    t = make_confirmations_token("dev-003", "reset")
    future = time.time() + 200
    monkeypatch.setattr(c.time, "time", lambda: future)
    assert valid_confirmation_token(t, "dev-003", "reset") is False

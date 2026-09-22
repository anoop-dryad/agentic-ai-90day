import time

from canopy_agent.confirmation import make_confirmation_token, valid_confirmation_token


def test_valid_token_passes():
    t = make_confirmation_token("dev-003", "reset")
    assert valid_confirmation_token(t, "dev-003", "reset") is True


def test_wrong_command_fails():
    t = make_confirmation_token("dev-003", "reset")
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
    t = make_confirmation_token("dev-003", "reset")
    # fast-forward past TTL
    import canopy_agent.confirmation as c

    monkeypatch.setattr(c.time, "time", lambda: time.time() + 200)
    assert valid_confirmation_token(t, "dev-003", "reset") is False

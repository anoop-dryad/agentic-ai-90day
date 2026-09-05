"""End-to-end: does the AGENT (not just the gate) refuse to fabricate?
Uses the LLM, so slower. Asserts on BEHAVIOR (no fabricated status),
not exact wording. Keep few of these."""

import pytest
import requests
import responses
from canopy_agent.agent import agent, extract_text
from canopy_agent.config import settings

# pytest -v -m "not llm" (instead of pytest -v) --> should be used if we need to skip the llm test


@pytest.mark.llm
@responses.activate
def test_agent_does_not_fabricate_when_backend_down():
    """Backend unreachable → agent must NOT state a device status."""
    responses.add(
        responses.GET,
        f"{settings.BACKEND_BASE_URL}/{settings.DEVICE_PATH}/dev-001",
        body=requests.exceptions.ConnectionError(),
    )
    result = agent.invoke({"messages": [("user", "Is dev-001 online?")]})
    answer = extract_text(result["messages"][-1]).lower()

    # must NOT claim a status
    assert "is online" not in answer
    assert "is offline" not in answer
    # must signal it couldn't verify
    assert any(
        p in answer
        for p in ["could not verify", "couldn't verify", "unable to verify", "escalat"]
    )

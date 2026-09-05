"""RAG grounding-gate tests."""

from canopy_agent.rag import build_index, search_docs_gated


def setup_module():
    build_index()


def test_status_question_is_grounded():
    res = search_docs_gated("what does inactive status mean?")
    assert res.grounded is True
    assert any("inactive" in c.lower() for c in res.chunks)


def test_connectivity_question_is_grounded():
    r = search_docs_gated("how do I fix connectivity issues?")
    assert r.grounded is True
    assert any("connectivity" in c.lower() or "gateway" in c.lower() for c in r.chunks)


def test_calibration_question_is_grounded():
    r = search_docs_gated("what is calibration mode?")
    assert r.grounded is True
    assert any("calibrat" in c.lower() for c in r.chunks)


def test_offtopic_question_not_grounded():
    """Off-topic → grounded=False, honest refusal path. The hallucination guard."""
    r = search_docs_gated("what is the airspeed velocity of a swallow?")
    assert r.grounded is False
    assert r.chunks == []

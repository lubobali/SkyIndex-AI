"""Tests for the optional summary layer (S1).

The summarizer is additive by design: every failure path must return None so
the search endpoint keeps serving ranked results. Most of these tests exist to
pin that property, because a summarizer that can take search down is worse
than no summarizer at all.
"""

from __future__ import annotations

import pytest

import rag

pytestmark = pytest.mark.fast


def make_result(**overrides) -> dict:
    result = {
        "location": "Cook County",
        "event": "Flash Flood Warning",
        "headline": "Flash Flood Warning issued...",
        "source_type": "alert",
        "chunk_text": "Rivers are rising fast. Move to higher ground.",
        "similarity": 0.81,
    }
    result.update(overrides)
    return result


# --------------------------------------------------------------------------
# context building
# --------------------------------------------------------------------------


def test_context_numbers_passages_for_citation():
    context = rag.build_context([make_result(), make_result(location="DuPage County")])
    assert "[1]" in context
    assert "[2]" in context


def test_context_includes_location_and_event():
    context = rag.build_context([make_result()])
    assert "Cook County" in context
    assert "Flash Flood Warning" in context
    assert "Rivers are rising fast." in context


def test_context_collapses_whitespace():
    """NWS text is hard-wrapped at ~68 columns. Left alone, the newlines eat
    context budget and fragment the prose the model reads."""
    result = make_result(chunk_text="Rivers are\nrising\n\nfast.")
    assert "Rivers are rising fast." in rag.build_context([result])


def test_context_is_capped_by_total_characters():
    """Capping by result count would be a poor proxy - a one-line forecast and
    a full warning differ in length by an order of magnitude."""
    results = [make_result(chunk_text="x" * 500) for _ in range(20)]
    context = rag.build_context(results, max_chars=1000)
    assert len(context) <= 1200  # cap plus the final entry's header


def test_context_of_nothing_is_empty():
    assert rag.build_context([]) == ""


# --------------------------------------------------------------------------
# graceful degradation - the property that matters most
# --------------------------------------------------------------------------


def test_no_results_means_no_model_call(monkeypatch):
    """Calling the model with no context invites it to invent weather."""
    monkeypatch.setattr(
        rag, "_credentials", lambda: pytest.fail("must not authenticate with no results")
    )
    assert rag.summarize("flooding", []) is None


def test_missing_credentials_returns_none(monkeypatch):
    monkeypatch.setattr(rag, "_credentials", lambda: (None, None))
    assert rag.summarize("flooding", [make_result()]) is None


def test_transport_failure_returns_none(monkeypatch):
    monkeypatch.setattr(rag, "_credentials", lambda: ("https://example.com", "tok"))

    import requests

    def _explode(*args, **kwargs):
        raise requests.ConnectionError("gateway unreachable")

    monkeypatch.setattr(requests, "post", _explode)
    assert rag.summarize("flooding", [make_result()]) is None


def test_http_error_returns_none(monkeypatch):
    monkeypatch.setattr(rag, "_credentials", lambda: ("https://example.com", "tok"))
    _stub_post(monkeypatch, status_code=403, payload={"error": "forbidden"})
    assert rag.summarize("flooding", [make_result()]) is None


def test_empty_endpoint_name_disables_summaries(monkeypatch):
    monkeypatch.setattr(
        rag, "_credentials", lambda: pytest.fail("must not authenticate when disabled")
    )
    assert rag.summarize("flooding", [make_result()], endpoint="") is None


# --------------------------------------------------------------------------
# the happy path and response shapes
# --------------------------------------------------------------------------


def _stub_post(monkeypatch, status_code=200, payload=None, capture=None):
    import requests

    class FakeResponse:
        def __init__(self):
            self.status_code = status_code

        def json(self):
            return payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(f"{self.status_code}")

    def _post(url, headers=None, json=None, timeout=None):
        if capture is not None:
            capture.update({"url": url, "headers": headers or {}, "body": json or {}})
        return FakeResponse()

    monkeypatch.setattr(requests, "post", _post)


def test_summary_is_returned_from_a_string_content(monkeypatch):
    monkeypatch.setattr(rag, "_credentials", lambda: ("https://example.com", "tok"))
    _stub_post(
        monkeypatch,
        payload={"choices": [{"message": {"content": "Flooding is expected [1]."}}]},
    )
    assert rag.summarize("flooding", [make_result()]) == "Flooding is expected [1]."


def test_summary_is_returned_from_block_style_content(monkeypatch):
    """Claude models return content as a list of typed blocks; the Gateway
    passes that shape through unchanged."""
    monkeypatch.setattr(rag, "_credentials", lambda: ("https://example.com", "tok"))
    _stub_post(
        monkeypatch,
        payload={
            "choices": [
                {"message": {"content": [{"type": "text", "text": "Heat advisory in effect."}]}}
            ]
        },
    )
    assert rag.summarize("heat", [make_result()]) == "Heat advisory in effect."


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": []},
        {"choices": [{}]},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": ""}}]},
        {"choices": [{"message": {"content": "   "}}]},
    ],
)
def test_unusable_responses_become_none(monkeypatch, payload):
    monkeypatch.setattr(rag, "_credentials", lambda: ("https://example.com", "tok"))
    _stub_post(monkeypatch, payload=payload)
    assert rag.summarize("flooding", [make_result()]) is None


def test_request_targets_the_ai_gateway_with_the_configured_model(monkeypatch):
    monkeypatch.setattr(rag, "_credentials", lambda: ("https://workspace.example.com", "tok"))
    captured: dict = {}
    _stub_post(
        monkeypatch,
        payload={"choices": [{"message": {"content": "ok"}}]},
        capture=captured,
    )

    rag.summarize("flooding", [make_result()], endpoint="system.ai.claude-haiku-4-5")

    assert captured["url"] == "https://workspace.example.com" + rag.SUMMARY_GATEWAY_PATH
    assert captured["headers"]["Authorization"] == "Bearer tok"
    assert captured["body"]["model"] == "system.ai.claude-haiku-4-5"


def test_prompt_constrains_the_model_to_the_passages(monkeypatch):
    """The system prompt is the only thing standing between a retrieval demo
    and a model inventing a hurricane. Worth asserting it stays."""
    monkeypatch.setattr(rag, "_credentials", lambda: ("https://example.com", "tok"))
    captured: dict = {}
    _stub_post(monkeypatch, payload={"choices": [{"message": {"content": "ok"}}]}, capture=captured)

    rag.summarize("flooding", [make_result()])

    system_message = captured["body"]["messages"][0]
    assert system_message["role"] == "system"
    assert "ONLY" in system_message["content"]
    assert "Never add hazards" in system_message["content"]


def test_temperature_is_low(monkeypatch):
    monkeypatch.setattr(rag, "_credentials", lambda: ("https://example.com", "tok"))
    captured: dict = {}
    _stub_post(monkeypatch, payload={"choices": [{"message": {"content": "ok"}}]}, capture=captured)

    rag.summarize("flooding", [make_result()])
    assert captured["body"]["temperature"] <= 0.2


def test_the_user_message_carries_the_question_and_passages(monkeypatch):
    monkeypatch.setattr(rag, "_credentials", lambda: ("https://example.com", "tok"))
    captured: dict = {}
    _stub_post(monkeypatch, payload={"choices": [{"message": {"content": "ok"}}]}, capture=captured)

    rag.summarize("is there flooding", [make_result()])
    user_message = captured["body"]["messages"][1]["content"]

    assert "is there flooding" in user_message
    assert "Cook County" in user_message

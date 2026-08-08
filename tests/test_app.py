"""Tests for the Flask endpoints (R10, R18, R20-R23, S1, S4)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.fast


@pytest.fixture
def client(monkeypatch):
    import app as app_module

    app_module.app.config.update(TESTING=True)
    with app_module.app.test_client() as test_client:
        yield test_client


@pytest.fixture
def stub_search(monkeypatch):
    """Replace the embed + database round trip with a recorded call."""
    import app as app_module

    calls: list[dict] = []
    state = {"results": []}

    def _run_search(query, top_k, source_type):
        calls.append({"query": query, "top_k": top_k, "source_type": source_type})
        return state["results"]

    monkeypatch.setattr(app_module, "_run_search", _run_search)
    return type("Stub", (), {"calls": calls, "state": state})()


def make_result(**overrides) -> dict:
    result = {
        "document_id": "urn:oid:1",
        "location": "Cook County",
        "headline": "Flash Flood Warning issued...",
        "event": "Flash Flood Warning",
        "source_type": "alert",
        "severity": "Severe",
        "chunk_index": 0,
        "chunk_text": "Rivers are rising fast.",
        "narrative_text": "Rivers are rising fast.",
        "effective_at": None,
        "expires_at": None,
        "similarity": 0.8231,
    }
    result.update(overrides)
    return result


# --------------------------------------------------------------------------
# R18, R20 - POST /weather/search
# --------------------------------------------------------------------------


def test_search_returns_ranked_results(client, stub_search):
    stub_search.state["results"] = [make_result()]
    response = client.post("/weather/search", json={"query": "flooding"})

    assert response.status_code == 200
    body = response.get_json()
    assert body["count"] == 1

    hit = body["results"][0]
    for field in ("location", "headline", "chunk_text", "similarity"):
        assert field in hit, f"the response contract requires {field}"
    assert hit["similarity"] == 0.8231


def test_search_passes_the_query_through(client, stub_search):
    client.post("/weather/search", json={"query": "  flash flood risk  "})
    assert stub_search.calls[0]["query"] == "flash flood risk"


def test_search_on_an_empty_index_returns_200(client, stub_search):
    """R21 - nothing synced yet is a normal state, not an error."""
    stub_search.state["results"] = []
    response = client.post("/weather/search", json={"query": "flooding"})

    assert response.status_code == 200
    assert response.get_json() == {
        "query": "flooding",
        "top_k": 5,
        "source_type": None,
        "count": 0,
        "results": [],
    }


# --------------------------------------------------------------------------
# R22 - malformed input
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"query": ""},
        {"query": "   "},
        {"query": None},
        {"query": 42},
        {"query": ["flooding"]},
        {"query": "x" * 1001},
    ],
)
def test_malformed_query_is_rejected_with_400(client, stub_search, body):
    response = client.post("/weather/search", json=body)
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_a_request_with_no_json_body_at_all_is_rejected(client, stub_search):
    response = client.post("/weather/search")
    assert response.status_code == 400


def test_errors_are_json_not_html(client, stub_search):
    """The UI parses every response as JSON; an HTML error page would surface
    as a parse error and hide the real message."""
    response = client.post("/weather/search", json={})
    assert response.content_type.startswith("application/json")


# --------------------------------------------------------------------------
# R23 - top_k bounds
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "requested,expected",
    [(None, 5), (1, 1), (3, 3), (20, 20), (21, 20), (500, 20), (0, 1), (-4, 1), ("7", 7)],
)
def test_top_k_is_clamped(client, stub_search, requested, expected):
    body = {"query": "flooding"}
    if requested is not None:
        body["top_k"] = requested

    client.post("/weather/search", json=body)
    assert stub_search.calls[0]["top_k"] == expected


@pytest.mark.parametrize("bad", ["abc", [], {}, True])
def test_non_numeric_top_k_is_rejected(client, stub_search, bad):
    response = client.post("/weather/search", json={"query": "flooding", "top_k": bad})
    assert response.status_code == 400


# --------------------------------------------------------------------------
# S4 - source_type filtering
# --------------------------------------------------------------------------


@pytest.mark.parametrize("source_type", ["alert", "forecast"])
def test_search_accepts_a_source_type_filter(client, stub_search, source_type):
    client.post("/weather/search", json={"query": "rain", "source_type": source_type})
    assert stub_search.calls[0]["source_type"] == source_type


@pytest.mark.parametrize("value", [None, "", "all", "any"])
def test_absent_or_all_source_type_searches_everything(client, stub_search, value):
    body = {"query": "rain"}
    if value is not None:
        body["source_type"] = value

    client.post("/weather/search", json=body)
    assert stub_search.calls[0]["source_type"] is None


def test_unknown_source_type_is_rejected(client, stub_search):
    response = client.post("/weather/search", json={"query": "rain", "source_type": "tsunami"})
    assert response.status_code == 400
    assert "tsunami" in response.get_json()["error"]


# --------------------------------------------------------------------------
# S1 - GET variant with a generated summary
# --------------------------------------------------------------------------


def test_get_search_includes_a_summary(client, stub_search, monkeypatch):
    import app as app_module

    stub_search.state["results"] = [make_result()]
    monkeypatch.setattr(app_module.rag, "summarize", lambda q, r: "Flooding is expected.")

    body = client.get("/weather/search?query=flooding").get_json()
    assert body["summary"] == "Flooding is expected."
    assert body["count"] == 1


def test_get_search_survives_an_unavailable_summarizer(client, stub_search, monkeypatch):
    """The summary is additive. A serving outage must not take search down."""
    import app as app_module

    stub_search.state["results"] = [make_result()]
    monkeypatch.setattr(app_module.rag, "summarize", lambda q, r: None)

    response = client.get("/weather/search?query=flooding")
    assert response.status_code == 200
    body = response.get_json()
    assert body["summary"] is None
    assert body["count"] == 1, "results are still served without a summary"


def test_get_search_does_not_summarize_nothing(client, stub_search, monkeypatch):
    """No results means no context - calling the model would invite invention."""
    import app as app_module

    called: list = []
    monkeypatch.setattr(app_module.rag, "summarize", lambda q, r: called.append(1))
    stub_search.state["results"] = []

    body = client.get("/weather/search?query=flooding").get_json()
    assert called == []
    assert body["summary"] is None


def test_get_search_validates_its_query_too(client, stub_search):
    assert client.get("/weather/search").status_code == 400
    assert client.get("/weather/search?query=").status_code == 400


# --------------------------------------------------------------------------
# R10 - POST /weather/sync
# --------------------------------------------------------------------------


@pytest.fixture
def stub_sync(monkeypatch):
    import app as app_module
    from weather_client import WeatherDocument

    harvested: list[dict] = []
    written: list[list] = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.errors: list[str] = []

        def harvest(self, locations, limit=50, source_types=("alert", "forecast")):
            harvested.append(
                {"locations": list(locations), "limit": limit,
                 "source_types": list(source_types)}
            )
            return [
                WeatherDocument(
                    id=f"doc-{index}", location="Cook County", latitude=None, longitude=None,
                    source_type="alert" if index % 2 == 0 else "forecast",
                    event="Flood Watch", headline="h", narrative_text="text",
                    issued_at=None, effective_at=None, expires_at=None,
                    severity=None, payload={},
                )
                for index in range(4)
            ]

    monkeypatch.setattr(app_module, "NWSClient", FakeClient)
    monkeypatch.setattr(
        app_module.repository, "upsert_documents",
        lambda docs: (written.append(list(docs)), len(list(docs)))[1],
    )
    return type("Stub", (), {"harvested": harvested, "written": written, "client": FakeClient})()


def test_sync_returns_a_count(client, stub_sync):
    body = client.post("/weather/sync", json={"locations": ["Chicago, IL"]}).get_json()
    assert body["synced"] == 4


def test_sync_reports_counts_per_source_type(client, stub_sync):
    body = client.post("/weather/sync", json={"locations": ["Chicago, IL"]}).get_json()
    assert body["by_source_type"] == {"alert": 2, "forecast": 2}


def test_sync_uses_default_locations_when_none_given(client, stub_sync):
    import app as app_module

    client.post("/weather/sync", json={})
    assert stub_sync.harvested[0]["locations"] == app_module.DEFAULT_LOCATIONS


def test_sync_accepts_a_bare_string_location(client, stub_sync):
    client.post("/weather/sync", json={"locations": "Chicago, IL"})
    assert stub_sync.harvested[0]["locations"] == ["Chicago, IL"]


@pytest.mark.parametrize(
    "requested,expected", [(None, 50), (10, 10), (0, 1), (9999, 200), ("25", 25)]
)
def test_sync_limit_is_clamped(client, stub_sync, requested, expected):
    body = {"locations": ["Chicago, IL"]}
    if requested is not None:
        body["limit"] = requested

    client.post("/weather/sync", json=body)
    assert stub_sync.harvested[0]["limit"] == expected


def test_sync_rejects_an_empty_location_list(client, stub_sync):
    response = client.post("/weather/sync", json={"locations": []})
    assert response.status_code == 400


def test_sync_rejects_too_many_locations(client, stub_sync):
    """Each location costs at least two upstream calls against a public API
    that asks for reasonable use."""
    response = client.post("/weather/sync", json={"locations": [f"City{i}, IL" for i in range(30)]})
    assert response.status_code == 400


def test_sync_can_restrict_source_types(client, stub_sync):
    client.post("/weather/sync", json={"locations": ["Chicago, IL"], "source_types": ["alert"]})
    assert stub_sync.harvested[0]["source_types"] == ["alert"]


def test_sync_rejects_an_unknown_source_type(client, stub_sync):
    response = client.post(
        "/weather/sync", json={"locations": ["Chicago, IL"], "source_types": ["blizzard"]}
    )
    assert response.status_code == 400


def test_sync_surfaces_per_location_failures(client, stub_sync, monkeypatch):
    """A partial failure must be visible. "synced: 40" reads like success even
    when half the batch failed."""
    import app as app_module

    class FailingClient(stub_sync.client):
        def harvest(self, locations, limit=50, source_types=("alert", "forecast")):
            documents = super().harvest(locations, limit, source_types)
            self.errors = ["99.0,99.0: 404 Client Error"]
            return documents

    monkeypatch.setattr(app_module, "NWSClient", FailingClient)
    body = client.post("/weather/sync", json={"locations": ["99.0,99.0"]}).get_json()
    assert body["errors"] == ["99.0,99.0: 404 Client Error"]


def test_sync_with_no_errors_omits_the_error_key(client, stub_sync):
    body = client.post("/weather/sync", json={"locations": ["Chicago, IL"]}).get_json()
    assert "errors" not in body


# --------------------------------------------------------------------------
# health
# --------------------------------------------------------------------------


def test_healthz_reports_ok_when_the_database_answers(client, monkeypatch):
    import app as app_module

    monkeypatch.setattr(app_module.lakebase, "healthcheck", lambda: True)
    monkeypatch.setattr(app_module.repository, "stats", lambda: {"documents": 111})

    body = client.get("/healthz").get_json()
    assert body["status"] == "ok"
    assert body["counts"]["documents"] == 111


def test_healthz_reports_degraded_when_the_database_is_down(client, monkeypatch):
    import app as app_module

    monkeypatch.setattr(app_module.lakebase, "healthcheck", lambda: False)
    response = client.get("/healthz")

    assert response.status_code == 503
    assert response.get_json()["status"] == "degraded"


def test_healthz_does_not_load_the_embedding_model(client, monkeypatch):
    """Loading the model takes seconds. A health check that waits on it would
    report unhealthy for an app that is simply cold."""
    import app as app_module

    monkeypatch.setattr(app_module.lakebase, "healthcheck", lambda: True)
    monkeypatch.setattr(app_module.repository, "stats", lambda: {})
    monkeypatch.setattr(
        app_module.embeddings, "get_encoder",
        lambda *a, **k: pytest.fail("healthz must not load the model"),
    )
    assert client.get("/healthz").status_code == 200

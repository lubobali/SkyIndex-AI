"""Tests for the NWS harvest layer (R1-R8).

Written before weather_client.py exists. The captured fixtures in
tests/fixtures/ are real api.weather.gov responses, so these assertions
describe the shape the service actually returns.
"""

from __future__ import annotations

import hashlib

import pytest
import requests

from conftest import FakeResponse, FakeSession
from weather_client import (
    GeocodeError,
    LocationParseError,
    NWSClient,
    ResolvedLocation,
    parse_location,
)

pytestmark = pytest.mark.fast


def make_client(session, **kwargs) -> NWSClient:
    kwargs.setdefault("user_agent", "(SkyIndex-AI test, test@example.com)")
    kwargs.setdefault("max_requests_per_second", 0)  # no pacing in unit tests
    return NWSClient(session=session, **kwargs)


# --------------------------------------------------------------------------
# R2 - location parsing accepts both "City, ST" and "lat,lon"
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,lat,lon",
    [
        ("41.8781,-87.6298", 41.8781, -87.6298),
        ("41.8781, -87.6298", 41.8781, -87.6298),
        ("  41.8781 , -87.6298  ", 41.8781, -87.6298),
        ("32.7157,-117.1611", 32.7157, -117.1611),
        ("40,-105", 40.0, -105.0),
    ],
)
def test_parse_location_accepts_latlon_forms(raw, lat, lon):
    parsed = parse_location(raw)
    assert parsed.is_coordinates
    assert parsed.latitude == pytest.approx(lat)
    assert parsed.longitude == pytest.approx(lon)


@pytest.mark.parametrize(
    "raw,city,state",
    [
        ("Chicago, IL", "Chicago", "IL"),
        ("Chicago,IL", "Chicago", "IL"),
        ("  austin ,  tx  ", "austin", "TX"),
        ("Salt Lake City, Utah", "Salt Lake City", "UT"),
    ],
)
def test_parse_location_accepts_city_state_forms(raw, city, state):
    parsed = parse_location(raw)
    assert not parsed.is_coordinates
    assert parsed.city == city
    assert parsed.state == state


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "Chicago",  # no state - too ambiguous to geocode reliably
        "Chicago, ZZ",  # not a US state
        "91.0,-87.0",  # latitude out of range
        "41.0,-181.0",  # longitude out of range
        "not a location at all",
    ],
)
def test_parse_location_rejects_unusable_input(raw):
    with pytest.raises(LocationParseError):
        parse_location(raw)


def test_parse_location_rejects_non_string():
    with pytest.raises(LocationParseError):
        parse_location(None)


# --------------------------------------------------------------------------
# R1, R7 - gridpoint resolution and the required User-Agent
# --------------------------------------------------------------------------


def test_client_requires_a_user_agent():
    """The NWS API rejects requests without one. Fail at construction with a
    clear message rather than at the first call with an opaque 403."""
    with pytest.raises(ValueError, match="[Uu]ser-[Aa]gent"):
        NWSClient(session=FakeSession(), user_agent="")


def test_user_agent_is_sent_on_every_request(nws_session):
    client = make_client(nws_session)
    client.resolve("41.8781,-87.6298")
    assert "User-Agent" in nws_session.headers
    assert "SkyIndex-AI" in nws_session.headers["User-Agent"]


def test_resolve_coordinates_skips_the_geocoder(nws_session):
    """A lat/lon location needs exactly one call - /points. Geocoding it
    would be a wasted round trip against a third-party service."""
    client = make_client(nws_session)
    location = client.resolve("41.8781,-87.6298")

    assert isinstance(location, ResolvedLocation)
    assert len(nws_session.calls) == 1
    assert "/points/41.8781,-87.6298" in nws_session.urls[0]


def test_resolve_coordinates_reads_grid_from_points_response(nws_session):
    client = make_client(nws_session)
    location = client.resolve("41.8781,-87.6298")

    assert location.office == "LOT"
    assert location.grid_x == 76
    assert location.grid_y == 73
    # The state comes from the NWS response, not from the input string - it is
    # what the alerts endpoint is queried with, so it has to be NWS's own view.
    assert location.state == "IL"
    assert location.label == "Chicago, IL"


def test_resolve_city_state_geocodes_then_resolves_grid(nws_session):
    client = make_client(nws_session)
    location = client.resolve("Chicago, IL")

    assert "geocoding-api.open-meteo.com" in nws_session.urls[0]
    assert "/points/" in nws_session.urls[1]
    assert location.office == "LOT"
    assert location.latitude == pytest.approx(41.85003)


def test_resolve_city_state_picks_the_match_in_the_requested_state():
    """"Austin" exists in five states. The one in TX is the one asked for."""
    session = FakeSession(
        {
            "geocoding-api.open-meteo.com": FakeResponse(
                {
                    "results": [
                        {"name": "Austin", "admin1": "Minnesota", "country_code": "US",
                         "latitude": 43.66663, "longitude": -92.97464},
                        {"name": "Austin", "admin1": "Texas", "country_code": "US",
                         "latitude": 30.26715, "longitude": -97.74306},
                    ]
                }
            ),
            "/points/": FakeResponse(
                {
                    "properties": {
                        "gridId": "EWX", "gridX": 156, "gridY": 91,
                        "relativeLocation": {"properties": {"city": "Austin", "state": "TX"}},
                    }
                }
            ),
        }
    )
    client = make_client(session)
    client.resolve("Austin, TX")

    points_url = [u for u in session.urls if "/points/" in u][0]
    assert "30.2672" in points_url
    assert "-97.7431" in points_url


def test_coordinates_are_rounded_to_four_decimals(nws_session):
    """api.weather.gov answers 301 for /points with more than four decimal
    places and 200 at four. Rounding here spends a rounding error measured in
    centimetres to avoid a redirect round trip on every single location."""
    client = make_client(nws_session)
    client.resolve("41.87811111,-87.62981111")
    assert "/points/41.8781,-87.6298" in nws_session.urls[0]


def test_resolve_city_state_ignores_non_us_results():
    session = FakeSession(
        {
            "geocoding-api.open-meteo.com": FakeResponse(
                {"results": [{"name": "Austin", "admin1": "Texas",
                              "country_code": "CA", "latitude": 1.0, "longitude": 2.0}]}
            )
        }
    )
    with pytest.raises(GeocodeError, match="Austin, TX"):
        make_client(session).resolve("Austin, TX")


def test_resolve_city_state_raises_when_geocoder_finds_nothing():
    session = FakeSession({"geocoding-api.open-meteo.com": FakeResponse({"results": []})})
    with pytest.raises(GeocodeError):
        make_client(session).resolve("Nowheresville, IL")


# --------------------------------------------------------------------------
# R3, R5, R6 - alert normalization
# --------------------------------------------------------------------------


def test_alert_documents_normalize_real_payload(nws_session, alerts_ca):
    client = make_client(nws_session)
    location = client.resolve("41.8781,-87.6298")
    docs = client.alert_documents(location, alerts_ca["features"])

    assert docs, "fixture contains alerts, so normalization must produce documents"
    doc = docs[0]
    assert doc.source_type == "alert"
    assert doc.event  # e.g. "Extreme Heat Warning"
    assert doc.headline
    assert doc.narrative_text
    assert doc.issued_at
    assert doc.severity
    assert doc.payload["properties"]["event"] == doc.event


def test_alert_id_is_the_nws_urn(nws_session, alerts_ca):
    """The alert's own id is already globally unique and issuer-versioned.
    Deriving our own would throw that away."""
    client = make_client(nws_session)
    location = client.resolve("41.8781,-87.6298")
    docs = client.alert_documents(location, alerts_ca["features"])

    ids = {doc.id for doc in docs}
    assert ids == {f["properties"]["id"] for f in alerts_ca["features"]}
    assert len(ids) == len(docs), "ids must be unique across a single fetch"


def test_alert_narrative_joins_description_and_instruction(nws_session):
    """`description` says what is happening, `instruction` says what to do.
    They are separate fields and both are worth retrieving on."""
    client = make_client(nws_session)
    location = client.resolve("41.8781,-87.6298")
    feature = {
        "properties": {
            "id": "urn:oid:test.1",
            "event": "Flash Flood Warning",
            "headline": "Flash Flood Warning issued...",
            "description": "Rivers are rising fast.",
            "instruction": "Move to higher ground.",
            "severity": "Severe",
            "areaDesc": "Cook County",
            "sent": "2026-08-07T12:00:00-05:00",
            "effective": "2026-08-07T12:00:00-05:00",
            "expires": "2026-08-08T00:00:00-05:00",
        }
    }
    doc = client.alert_documents(location, [feature])[0]

    assert "Rivers are rising fast." in doc.narrative_text
    assert "Move to higher ground." in doc.narrative_text


def test_alert_with_no_instruction_still_produces_a_document(nws_session):
    """Many advisories carry a null instruction. That is normal, not an error."""
    client = make_client(nws_session)
    location = client.resolve("41.8781,-87.6298")
    feature = {
        "properties": {
            "id": "urn:oid:test.2",
            "event": "Air Quality Alert",
            "description": "Ozone levels are elevated.",
            "instruction": None,
            "sent": "2026-08-07T12:00:00-05:00",
        }
    }
    doc = client.alert_documents(location, [feature])[0]
    assert doc.narrative_text == "Ozone levels are elevated."


def test_alert_with_no_narrative_text_is_skipped(nws_session):
    """A document with nothing to embed would become a zero-information
    vector that still competes for a slot in every search result."""
    client = make_client(nws_session)
    location = client.resolve("41.8781,-87.6298")
    features = [
        {"properties": {"id": "urn:oid:test.3", "event": "Test",
                        "description": None, "instruction": None,
                        "sent": "2026-08-07T12:00:00-05:00"}},
        {"properties": {"id": "urn:oid:test.4", "event": "Test",
                        "description": "   ", "instruction": "  ",
                        "sent": "2026-08-07T12:00:00-05:00"}},
    ]
    assert client.alert_documents(location, features) == []


def test_alert_area_desc_is_preferred_over_the_grid_label(nws_session):
    """An alert covers named counties, which is more precise than the
    gridpoint city we happened to resolve it through."""
    client = make_client(nws_session)
    location = client.resolve("41.8781,-87.6298")
    feature = {
        "properties": {
            "id": "urn:oid:test.5",
            "event": "Flood Watch",
            "description": "Flooding possible.",
            "areaDesc": "Cook County; DuPage County",
            "sent": "2026-08-07T12:00:00-05:00",
        }
    }
    doc = client.alert_documents(location, [feature])[0]
    assert doc.location == "Cook County; DuPage County"


# --------------------------------------------------------------------------
# R4, R5, R6 - forecast normalization
# --------------------------------------------------------------------------


def test_forecast_documents_normalize_real_payload(nws_session, forecast_lot):
    client = make_client(nws_session)
    location = client.resolve("41.8781,-87.6298")
    docs = client.forecast_documents(location, forecast_lot)

    assert len(docs) == len(forecast_lot["properties"]["periods"])
    doc = docs[0]
    assert doc.source_type == "forecast"
    assert doc.location == "Chicago, IL"
    assert doc.event == forecast_lot["properties"]["periods"][0]["name"]
    assert doc.narrative_text == forecast_lot["properties"]["periods"][0]["detailedForecast"]
    assert doc.headline == forecast_lot["properties"]["periods"][0]["shortForecast"]
    assert doc.severity is None
    assert doc.effective_at == forecast_lot["properties"]["periods"][0]["startTime"]
    assert doc.expires_at == forecast_lot["properties"]["periods"][0]["endTime"]


def test_forecast_ids_are_stable_across_repeated_fetches(nws_session, forecast_lot):
    """Re-running sync against an unchanged forecast must update the same
    rows, not insert a second copy of every period."""
    client = make_client(nws_session)
    location = client.resolve("41.8781,-87.6298")

    first = [d.id for d in client.forecast_documents(location, forecast_lot)]
    second = [d.id for d in client.forecast_documents(location, forecast_lot)]
    assert first == second


def test_forecast_ids_are_distinct_per_period(nws_session, forecast_lot):
    client = make_client(nws_session)
    location = client.resolve("41.8781,-87.6298")
    docs = client.forecast_documents(location, forecast_lot)
    assert len({d.id for d in docs}) == len(docs)


def test_forecast_id_is_keyed_on_the_time_window_not_the_period_name(nws_session):
    """NWS renames periods as the day advances - the 15:00-18:00 window is
    "This Afternoon" now and part of "Today" in an earlier issuance. Keying
    on the name would duplicate the same forecast under two ids."""
    client = make_client(nws_session)
    location = client.resolve("41.8781,-87.6298")

    window = {"startTime": "2026-08-07T15:00:00-05:00", "endTime": "2026-08-07T18:00:00-05:00"}
    as_afternoon = {"number": 1, "name": "This Afternoon",
                    "detailedForecast": "Mostly sunny.", "shortForecast": "Mostly Sunny", **window}
    as_today = {"number": 1, "name": "Today",
                "detailedForecast": "Mostly sunny.", "shortForecast": "Mostly Sunny", **window}

    first = client.forecast_documents(location, {"properties": {"periods": [as_afternoon]}})[0]
    second = client.forecast_documents(location, {"properties": {"periods": [as_today]}})[0]
    assert first.id == second.id


def test_forecast_period_with_no_detail_is_skipped(nws_session):
    client = make_client(nws_session)
    location = client.resolve("41.8781,-87.6298")
    periods = [{"number": 1, "name": "Tonight", "detailedForecast": "",
                "startTime": "2026-08-07T18:00:00-05:00",
                "endTime": "2026-08-08T06:00:00-05:00"}]
    assert client.forecast_documents(location, {"properties": {"periods": periods}}) == []


# --------------------------------------------------------------------------
# content_hash - the field that makes re-embedding correct
# --------------------------------------------------------------------------


def test_content_hash_is_sha256_of_the_embedded_text(nws_session, alerts_ca):
    client = make_client(nws_session)
    location = client.resolve("41.8781,-87.6298")
    doc = client.alert_documents(location, alerts_ca["features"])[0]
    expected = hashlib.sha256(doc.narrative_text.encode("utf-8")).hexdigest()
    assert doc.content_hash == expected


def test_content_hash_changes_when_an_alert_is_amended(nws_session):
    """Same alert id, amended text. The hash is what lets the embedding job
    notice, so the stale vector is replaced instead of silently kept."""
    client = make_client(nws_session)
    location = client.resolve("41.8781,-87.6298")
    base = {"id": "urn:oid:test.6", "event": "Flood Watch",
            "sent": "2026-08-07T12:00:00-05:00"}

    original = client.alert_documents(
        location, [{"properties": {**base, "description": "Flooding possible."}}]
    )[0]
    amended = client.alert_documents(
        location, [{"properties": {**base, "description": "Flooding now likely."}}]
    )[0]

    assert original.id == amended.id
    assert original.content_hash != amended.content_hash


# --------------------------------------------------------------------------
# R8 - pacing and retries
# --------------------------------------------------------------------------


def test_requests_are_paced_to_the_configured_rate():
    clock = {"now": 0.0}
    slept: list[float] = []
    session = FakeSession({"/points/": FakeResponse(
        {"properties": {"gridId": "LOT", "gridX": 1, "gridY": 2,
                        "relativeLocation": {"properties": {"city": "X", "state": "IL"}}}})})

    client = NWSClient(
        session=session,
        user_agent="(test, test@example.com)",
        max_requests_per_second=2,  # one request every 0.5s
        sleep=lambda s: (slept.append(s), clock.__setitem__("now", clock["now"] + s)),
        clock=lambda: clock["now"],
    )
    client.resolve("41.0,-87.0")
    client.resolve("42.0,-88.0")

    assert slept, "the second call must wait for the rate limiter"
    assert slept[0] == pytest.approx(0.5, abs=0.01)


def test_no_pacing_delay_on_the_very_first_request():
    slept: list[float] = []
    session = FakeSession({"/points/": FakeResponse(
        {"properties": {"gridId": "LOT", "gridX": 1, "gridY": 2,
                        "relativeLocation": {"properties": {"city": "X", "state": "IL"}}}})})
    client = NWSClient(session=session, user_agent="(test, t@e.com)",
                       max_requests_per_second=1,
                       sleep=lambda s: slept.append(s), clock=lambda: 0.0)
    client.resolve("41.0,-87.0")
    assert slept == []


def test_transient_server_error_is_retried():
    attempts = {"n": 0}

    def flaky(url, params):
        attempts["n"] += 1
        if attempts["n"] < 3:
            return FakeResponse(status_code=503)
        return FakeResponse(
            {"properties": {"gridId": "LOT", "gridX": 1, "gridY": 2,
                            "relativeLocation": {"properties": {"city": "X", "state": "IL"}}}}
        )

    session = FakeSession({"/points/": flaky})
    client = make_client(session, max_retries=3, sleep=lambda s: None)
    location = client.resolve("41.0,-87.0")

    assert attempts["n"] == 3
    assert location.office == "LOT"


def test_client_error_is_not_retried():
    """A 404 from /points means the coordinates are outside NWS coverage.
    Retrying cannot change that and only burns the rate limit."""
    attempts = {"n": 0}

    def not_found(url, params):
        attempts["n"] += 1
        return FakeResponse(status_code=404)

    session = FakeSession({"/points/": not_found})
    client = make_client(session, max_retries=3, sleep=lambda s: None)

    with pytest.raises(requests.HTTPError):
        client.resolve("41.0,-87.0")
    assert attempts["n"] == 1


def test_retries_are_bounded():
    attempts = {"n": 0}

    def always_down(url, params):
        attempts["n"] += 1
        return FakeResponse(status_code=500)

    session = FakeSession({"/points/": always_down})
    client = make_client(session, max_retries=2, sleep=lambda s: None)

    with pytest.raises(requests.HTTPError):
        client.resolve("41.0,-87.0")
    assert attempts["n"] == 2


# --------------------------------------------------------------------------
# harvest() - the whole Part 1 path
# --------------------------------------------------------------------------


def test_harvest_returns_both_source_types(nws_session):
    docs = make_client(nws_session).harvest(["41.8781,-87.6298"])
    assert {d.source_type for d in docs} == {"alert", "forecast"}


def test_harvest_can_restrict_to_one_source_type(nws_session):
    docs = make_client(nws_session).harvest(["41.8781,-87.6298"], source_types=["forecast"])
    assert {d.source_type for d in docs} == {"forecast"}
    assert not any("/alerts/active" in u for u in nws_session.urls)


def test_harvest_limit_caps_documents_per_location_and_source(nws_session):
    docs = make_client(nws_session).harvest(["41.8781,-87.6298"], limit=2)
    for source_type in ("alert", "forecast"):
        assert len([d for d in docs if d.source_type == source_type]) <= 2


def test_harvest_deduplicates_ids_across_locations(nws_session):
    """Two nearby locations share a forecast office and can surface the same
    alert. Emitting it twice would fail the upsert's ON CONFLICT expectations
    within a single batch."""
    docs = make_client(nws_session).harvest(["41.8781,-87.6298", "41.9,-87.7"])
    ids = [d.id for d in docs]
    assert len(ids) == len(set(ids))


def test_harvest_skips_a_failing_location_and_keeps_going(nws_session, points_chicago):
    """One bad location in a batch of ten must not lose the other nine."""
    session = FakeSession(dict(nws_session.routes))
    session.route(
        "/points/99.0,99.0",
        requests.HTTPError("404 not found", response=FakeResponse(status_code=404)),
    )
    client = make_client(session)
    result = client.harvest(["99.0,99.0", "41.8781,-87.6298"])

    assert result, "the healthy location still produced documents"
    assert client.errors, "the failure is reported rather than swallowed silently"
    assert "99.0,99.0" in client.errors[0]


def test_harvest_reports_zero_documents_without_raising(points_chicago, forecast_lot):
    """A quiet state has no active alerts. Empty is a valid answer."""
    session = FakeSession(
        {
            "/points/": FakeResponse(points_chicago),
            "/gridpoints/": FakeResponse({"properties": {"periods": []}}),
            "/alerts/active": FakeResponse({"features": []}),
        }
    )
    assert make_client(session).harvest(["41.8781,-87.6298"]) == []


def test_alerts_are_queried_for_the_state_nws_reported(nws_session):
    make_client(nws_session).harvest(["41.8781,-87.6298"], source_types=["alert"])
    alert_calls = [(u, p) for u, p in nws_session.calls if "/alerts/active" in u]
    assert alert_calls[0][1]["area"] == "IL"


def test_alerts_active_is_not_sent_an_unsupported_limit_param(nws_session):
    """/alerts/active rejects `limit` with a 400 - the parameter only exists
    on /alerts. Capping happens client-side after the fetch instead.

    This is worth a regression test because the failure is silent when read
    through a lenient parser: the 400 body has no "features" key, so naive
    code reports "no active alerts" for a state that has thirty."""
    make_client(nws_session).harvest(["41.8781,-87.6298"], limit=3, source_types=["alert"])
    alert_params = [p for u, p in nws_session.calls if "/alerts/active" in u][0]
    assert "limit" not in alert_params


def test_http_error_message_includes_the_api_problem_detail():
    """NWS returns a machine-readable problem document naming the offending
    parameter. Discarding it leaves only "400 Client Error", which says
    nothing about which parameter was wrong."""
    body = {
        "title": "Bad Request",
        "detail": "Bad Request",
        "parameterErrors": [
            {"parameter": "query.limit", "message": 'Query parameter "limit" is not recognized'}
        ],
    }
    session = FakeSession({"/alerts/active": FakeResponse(body, status_code=400)})
    client = make_client(session, max_retries=1, sleep=lambda s: None)
    location = ResolvedLocation("Chicago, IL", 41.8781, -87.6298, "IL", "LOT", 76, 73)

    with pytest.raises(requests.HTTPError) as excinfo:
        client.fetch_alerts(location)
    assert "limit" in str(excinfo.value)

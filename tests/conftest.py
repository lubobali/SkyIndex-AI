"""Shared test fixtures.

The only things mocked here are the genuine external boundaries: outbound
HTTP (api.weather.gov, the geocoder, the model serving endpoint) and the
Postgres wire protocol. Everything inside the project - normalization,
chunking, id derivation, SQL construction, request handling - runs for real.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# Importing app.py kicks off a background load of the real embedding model.
# That is right for a server and wrong for a test run: it downloads weights,
# burns seconds, and makes the suite depend on network and disk cache. Set
# before any test module imports app.
os.environ.setdefault("WARM_ENCODER", "false")

FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    """Load a captured api.weather.gov response.

    These are real recorded payloads rather than hand-written dicts, so a
    test passing means the normalizer handles the shape the API actually
    returns - including the fields nobody remembers, like an alert whose
    `instruction` is null.
    """
    return json.loads((FIXTURE_DIR / f"{name}.json").read_text())


@pytest.fixture
def points_chicago() -> dict:
    return load_fixture("points_chicago")


@pytest.fixture
def forecast_lot() -> dict:
    return load_fixture("forecast_lot")


@pytest.fixture
def alerts_ca() -> dict:
    return load_fixture("alerts_ca")


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, payload=None, status_code: int = 200, text: str = ""):
        self._payload = payload
        self.status_code = status_code
        self.text = text or json.dumps(payload if payload is not None else {})

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON payload")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code} error", response=self)


class FakeSession:
    """Routes outbound GETs to canned responses by URL substring.

    Records every call so tests can assert on how many requests were made and
    in what order - which is how the rate limiting and the "one /points call
    per location" behaviour get verified.
    """

    def __init__(self, routes: dict[str, object] | None = None):
        self.routes = dict(routes or {})
        self.calls: list[tuple[str, dict]] = []
        self.headers: dict[str, str] = {}

    def route(self, fragment: str, response) -> "FakeSession":
        self.routes[fragment] = response
        return self

    def get(self, url: str, params=None, timeout=None, headers=None):
        self.calls.append((url, dict(params or {})))
        # Longest fragment wins, so a test can register a specific override
        # (e.g. "/points/99.0,99.0" -> error) on top of a general route
        # ("/points/" -> success) regardless of insertion order.
        for fragment in sorted(self.routes, key=len, reverse=True):
            response = self.routes[fragment]
            if fragment in url:
                if callable(response):
                    response = response(url, params)
                if isinstance(response, Exception):
                    raise response
                return response
        raise AssertionError(f"No fake route registered for URL: {url}")

    @property
    def urls(self) -> list[str]:
        return [url for url, _ in self.calls]


@pytest.fixture
def fake_session() -> FakeSession:
    return FakeSession()


class FakeCursor:
    """Records SQL instead of executing it, and serves queued results.

    This verifies how queries are *built* - the joins, the casts, the ON
    CONFLICT targets, the parameter order. It cannot verify that the SQL is
    valid Postgres; that is what the `live` integration tests against a real
    Lakebase are for. Both layers are needed, and neither substitutes for
    the other.
    """

    def __init__(self, connection: "FakeConnection"):
        self._connection = connection
        self._result: list | None = None
        self.rowcount = -1

    def execute(self, sql, params=None):
        self._connection.executed.append((normalize_sql(sql), params))
        self._result = self._connection.pop_result()
        self.rowcount = len(self._result) if isinstance(self._result, list) else 0

    def fetchall(self):
        return self._result or []

    def fetchone(self):
        rows = self._result or []
        return rows[0] if rows else None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConnection:
    def __init__(self, results=None):
        self.executed: list[tuple[str, object]] = []
        self.results: list = list(results or [])
        self.committed = False
        self.rolled_back = False

    def cursor(self, *args, **kwargs):
        return FakeCursor(self)

    def pop_result(self):
        return self.results.pop(0) if self.results else []

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    @property
    def statements(self) -> list[str]:
        return [sql for sql, _ in self.executed]

    def find(self, *fragments: str) -> tuple[str, object]:
        """The first statement containing all the given fragments."""
        for sql, params in self.executed:
            if all(fragment.lower() in sql.lower() for fragment in fragments):
                return sql, params
        raise AssertionError(
            f"No statement matching {fragments}. Executed:\n"
            + "\n".join(self.statements)
        )


def normalize_sql(sql: str) -> str:
    """Collapse whitespace so assertions can look for phrases, not formatting."""
    return " ".join(str(sql).split())


class FakeDB:
    """Patches the repository's database boundary for a single test."""

    def __init__(self, monkeypatch, results=None):
        import lakebase
        import repository
        from contextlib import contextmanager

        self.connection = FakeConnection(results)
        self.execute_values_calls: list[dict] = []

        @contextmanager
        def _get_connection():
            yield self.connection

        monkeypatch.setattr(lakebase, "get_connection", _get_connection)

        def _execute_values(cursor, sql, argslist, template=None, page_size=100):
            rows = list(argslist)
            self.execute_values_calls.append(
                {
                    "sql": normalize_sql(sql),
                    "rows": rows,
                    "template": normalize_sql(template) if template else None,
                    "page_size": page_size,
                }
            )
            self.connection.executed.append((normalize_sql(sql), rows))
            cursor.rowcount = len(rows)

        monkeypatch.setattr(repository, "execute_values", _execute_values)

    def queue(self, *results) -> "FakeDB":
        self.connection.results.extend(results)
        return self

    @property
    def last_write(self) -> dict:
        assert self.execute_values_calls, "no batched write was issued"
        return self.execute_values_calls[-1]


@pytest.fixture
def fake_db(monkeypatch) -> FakeDB:
    return FakeDB(monkeypatch)


@pytest.fixture
def nws_session(points_chicago, forecast_lot, alerts_ca) -> FakeSession:
    """A session wired for the full happy path: geocode -> points -> data."""
    return FakeSession(
        {
            "geocoding-api.open-meteo.com": FakeResponse(
                {
                    "results": [
                        {
                            "name": "Chicago",
                            "admin1": "Illinois",
                            "country_code": "US",
                            "latitude": 41.85003,
                            "longitude": -87.65005,
                        }
                    ]
                }
            ),
            "/points/": FakeResponse(points_chicago),
            "/gridpoints/": FakeResponse(forecast_lot),
            "/alerts/active": FakeResponse(alerts_ca),
        }
    )

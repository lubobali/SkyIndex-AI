"""Harvest unstructured weather narrative from the National Weather Service.

api.weather.gov publishes two kinds of free text worth retrieving on:

  alerts    /alerts/active?area={state}
            `description` is 1000-2000 characters of WHAT / WHERE / WHEN /
            IMPACTS prose; `instruction` is a separate block of protective
            action text. Both are embedded, joined.

  forecasts /gridpoints/{office}/{x},{y}/forecast
            one `detailedForecast` narrative per period, roughly 14 periods
            covering a week.

Both normalize into the same WeatherDocument shape so a single vector index
serves both, filterable by source_type.

The API takes coordinates, not place names, so "Chicago, IL" is geocoded
first (Open-Meteo, no key) and the result handed to /points to get the
forecast office and grid square.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import requests

logger = logging.getLogger(__name__)

NWS_API_BASE_URL = os.environ.get("NWS_API_BASE_URL", "https://api.weather.gov")
GEOCODER_URL = os.environ.get(
    "GEOCODER_URL", "https://geocoding-api.open-meteo.com/v1/search"
)
DEFAULT_USER_AGENT = os.environ.get("NWS_USER_AGENT", "")
DEFAULT_RATE = float(os.environ.get("NWS_MAX_REQUESTS_PER_SECOND", "4"))

SOURCE_ALERT = "alert"
SOURCE_FORECAST = "forecast"
SOURCE_TYPES = (SOURCE_ALERT, SOURCE_FORECAST)

# api.weather.gov answers 301 for /points coordinates carrying more than four
# decimal places, and 200 at four. Rounding costs a few centimetres of
# precision - far below the resolution of a 2.5km forecast grid square - and
# saves a redirect round trip on every location resolved.
_COORD_PRECISION = 4

_COORD_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*$")

_US_STATES = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR",
    "CALIFORNIA": "CA", "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE",
    "DISTRICT OF COLUMBIA": "DC", "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI",
    "IDAHO": "ID", "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA",
    "KANSAS": "KS", "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME",
    "MARYLAND": "MD", "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN",
    "MISSISSIPPI": "MS", "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE",
    "NEVADA": "NV", "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM",
    "NEW YORK": "NY", "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH",
    "OKLAHOMA": "OK", "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI",
    "SOUTH CAROLINA": "SC", "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX",
    "UTAH": "UT", "VERMONT": "VT", "VIRGINIA": "VA", "WASHINGTON": "WA",
    "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY",
    # NWS covers these territories too, and they get real alerts.
    "PUERTO RICO": "PR", "GUAM": "GU", "AMERICAN SAMOA": "AS",
    "VIRGIN ISLANDS": "VI", "U.S. VIRGIN ISLANDS": "VI",
    "NORTHERN MARIANA ISLANDS": "MP",
}
_STATE_CODES = set(_US_STATES.values())


class WeatherClientError(RuntimeError):
    """Base class for harvest failures."""


class LocationParseError(WeatherClientError):
    """The location string is not a usable "City, ST" or "lat,lon"."""


class GeocodeError(WeatherClientError):
    """A place name could not be resolved to coordinates."""


@dataclass(frozen=True)
class ParsedLocation:
    """A location request, before it has touched the network."""

    raw: str
    is_coordinates: bool
    latitude: float | None = None
    longitude: float | None = None
    city: str | None = None
    state: str | None = None

    @property
    def label(self) -> str:
        if self.is_coordinates:
            return f"{self.latitude},{self.longitude}"
        return f"{self.city}, {self.state}"


@dataclass(frozen=True)
class ResolvedLocation:
    """A location pinned to an NWS forecast office and grid square."""

    label: str
    latitude: float
    longitude: float
    state: str
    office: str
    grid_x: int
    grid_y: int


@dataclass(frozen=True)
class WeatherDocument:
    """One retrievable weather narrative, ready to store and embed."""

    id: str
    location: str
    latitude: float | None
    longitude: float | None
    source_type: str
    event: str | None
    headline: str | None
    narrative_text: str
    issued_at: str | None
    effective_at: str | None
    expires_at: str | None
    severity: str | None
    payload: dict
    content_hash: str = field(default="", repr=False)

    def __post_init__(self):
        if not self.content_hash:
            object.__setattr__(self, "content_hash", content_hash(self.narrative_text))

    def as_row(self) -> dict:
        """Column-name-keyed dict matching weather_documents."""
        return {
            "id": self.id,
            "location": self.location,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "source_type": self.source_type,
            "event": self.event,
            "headline": self.headline,
            "narrative_text": self.narrative_text,
            "issued_at": self.issued_at,
            "effective_at": self.effective_at,
            "expires_at": self.expires_at,
            "severity": self.severity,
            "payload": self.payload,
            "content_hash": self.content_hash,
        }


def content_hash(text: str) -> str:
    """Fingerprint the exact text that gets embedded.

    NWS re-issues an alert under its original id with amended wording. Storing
    this hash on both the document and its vectors is what lets the embedding
    job tell "already embedded" from "embedded, but the text has since
    changed" - the difference between skipping work and serving stale text.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_state(value: str) -> str | None:
    """Map "IL" or "Illinois" to the two-letter code NWS uses."""
    candidate = value.strip().upper()
    if candidate in _STATE_CODES:
        return candidate
    return _US_STATES.get(candidate)


def parse_location(raw: Any) -> ParsedLocation:
    """Parse a user-supplied location into coordinates or city/state.

    Accepts "41.8781,-87.6298" and "Chicago, IL" (or "Chicago, Illinois").
    A bare city name is rejected: "Springfield" alone matches places in more
    than thirty states, and silently guessing one of them would return
    confident weather for the wrong part of the country.
    """
    if not isinstance(raw, str):
        raise LocationParseError(f"Location must be a string, got {type(raw).__name__}")

    text = raw.strip()
    if not text:
        raise LocationParseError("Location must not be empty")

    coords = _COORD_RE.match(text)
    if coords:
        latitude, longitude = float(coords.group(1)), float(coords.group(2))
        if not -90.0 <= latitude <= 90.0:
            raise LocationParseError(f"Latitude {latitude} is outside -90..90")
        if not -180.0 <= longitude <= 180.0:
            raise LocationParseError(f"Longitude {longitude} is outside -180..180")
        return ParsedLocation(
            raw=text,
            is_coordinates=True,
            latitude=round(latitude, _COORD_PRECISION),
            longitude=round(longitude, _COORD_PRECISION),
        )

    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 2 or not parts[0]:
        raise LocationParseError(
            f"Could not parse location {raw!r}. Use 'City, ST' or 'lat,lon'."
        )

    city, state_raw = parts
    state = _normalize_state(state_raw)
    if not state:
        raise LocationParseError(f"{state_raw!r} is not a US state (in {raw!r})")

    return ParsedLocation(raw=text, is_coordinates=False, city=city, state=state)


class _Pacer:
    """Spaces outbound requests to a fixed maximum rate.

    NWS does not publish a hard numeric limit; it asks for reasonable use and
    throttles abusive clients. Pacing deliberately rather than discovering the
    limit by being throttled.
    """

    def __init__(self, requests_per_second: float, sleep: Callable, clock: Callable):
        self.min_interval = 1.0 / requests_per_second if requests_per_second > 0 else 0.0
        self._sleep = sleep
        self._clock = clock
        self._last: float | None = None

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        now = self._clock()
        if self._last is not None:
            remaining = self.min_interval - (now - self._last)
            if remaining > 0:
                self._sleep(remaining)
                now = self._clock()
        self._last = now


class NWSClient:
    """Fetches and normalizes weather narrative from api.weather.gov."""

    def __init__(
        self,
        base_url: str | None = None,
        geocoder_url: str | None = None,
        user_agent: str | None = None,
        timeout: int = 30,
        max_requests_per_second: float | None = None,
        max_retries: int = 3,
        backoff: float = 0.5,
        session: Any | None = None,
        sleep: Callable = time.sleep,
        clock: Callable = time.monotonic,
    ):
        agent = DEFAULT_USER_AGENT if user_agent is None else user_agent
        if not agent or not agent.strip():
            raise ValueError(
                "api.weather.gov requires a descriptive User-Agent identifying "
                "the application and a contact address, e.g. "
                "'(SkyIndex-AI, you@example.com)'. Set NWS_USER_AGENT or pass "
                "user_agent=."
            )

        self.base_url = (base_url or NWS_API_BASE_URL).rstrip("/")
        self.geocoder_url = geocoder_url or GEOCODER_URL
        self.user_agent = agent.strip()
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.backoff = backoff
        self._sleep = sleep
        self.session = session if session is not None else requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent, "Accept": "application/geo+json"})

        rate = DEFAULT_RATE if max_requests_per_second is None else max_requests_per_second
        self._pacer = _Pacer(rate, sleep, clock)

        # Non-fatal per-location failures collected during harvest().
        self.errors: list[str] = []

    # -- HTTP ---------------------------------------------------------------

    def _get(self, url: str, params: dict | None = None) -> Any:
        """GET with pacing, bounded retries, and no retrying of 4xx.

        A 4xx is a statement about the request - coordinates outside NWS
        coverage, a malformed grid - and repeating it cannot change the
        answer. Only 5xx and transport errors are worth another attempt.
        """
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            self._pacer.wait()
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except requests.HTTPError as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and status < 500:
                    raise self._with_problem_detail(exc)
                last_error = exc
            except requests.RequestException as exc:
                last_error = exc

            if attempt < self.max_retries:
                delay = self.backoff * (2 ** (attempt - 1))
                logger.warning(
                    "%s failed (attempt %s/%s), retrying in %.1fs",
                    url, attempt, self.max_retries, delay,
                )
                self._sleep(delay)

        raise last_error  # type: ignore[misc]

    @staticmethod
    def _with_problem_detail(exc: requests.HTTPError) -> requests.HTTPError:
        """Fold the API's problem document into the exception message.

        NWS answers a bad request with an RFC 7807 body naming the offending
        parameter. Letting only "400 Client Error" escape throws away the one
        piece of information that identifies the mistake.
        """
        try:
            body = exc.response.json()
        except Exception:  # noqa: BLE001 - non-JSON error bodies are fine to ignore
            return exc

        details = [str(body[key]) for key in ("title", "detail") if body.get(key)]
        for parameter_error in body.get("parameterErrors") or []:
            details.append(
                f"{parameter_error.get('parameter')}: {parameter_error.get('message')}"
            )
        if not details:
            return exc

        enriched = requests.HTTPError(f"{exc} - {'; '.join(details)}", response=exc.response)
        return enriched

    # -- Resolution ---------------------------------------------------------

    def _geocode(self, parsed: ParsedLocation) -> tuple[float, float]:
        """Resolve "City, ST" to coordinates, honouring the requested state."""
        payload = self._get(
            self.geocoder_url,
            params={
                "name": parsed.city,
                "count": 10,
                "countryCode": "US",
                "language": "en",
                "format": "json",
            },
        )
        for result in payload.get("results") or []:
            if (result.get("country_code") or "").upper() != "US":
                continue
            if _normalize_state(result.get("admin1") or "") != parsed.state:
                continue
            return (
                round(float(result["latitude"]), _COORD_PRECISION),
                round(float(result["longitude"]), _COORD_PRECISION),
            )

        raise GeocodeError(f"No US place found matching {parsed.label}")

    def resolve(self, location: str) -> ResolvedLocation:
        """Resolve a location string to an NWS forecast office and grid square."""
        parsed = parse_location(location)

        if parsed.is_coordinates:
            latitude, longitude = parsed.latitude, parsed.longitude
        else:
            latitude, longitude = self._geocode(parsed)

        payload = self._get(f"{self.base_url}/points/{latitude},{longitude}")
        properties = payload["properties"]
        relative = (properties.get("relativeLocation") or {}).get("properties") or {}

        city = relative.get("city") or (parsed.city if not parsed.is_coordinates else None)
        # The state is read back from NWS rather than from the input, because
        # it is the key the alerts endpoint is queried with. A point just over
        # a state line belongs to the state NWS assigns it, not the one the
        # caller typed.
        state = (relative.get("state") or parsed.state or "").upper()
        label = f"{city}, {state}" if city and state else f"{latitude},{longitude}"

        return ResolvedLocation(
            label=label,
            latitude=float(latitude),
            longitude=float(longitude),
            state=state,
            office=properties["gridId"],
            grid_x=int(properties["gridX"]),
            grid_y=int(properties["gridY"]),
        )

    # -- Fetching -----------------------------------------------------------

    def fetch_alerts(self, location: ResolvedLocation) -> list[dict]:
        """Active alerts for this location's state.

        No `limit` parameter: /alerts/active rejects it with a 400 (it exists
        only on the archival /alerts endpoint). Capping is done by the caller
        after normalization. The failure mode this avoids is a quiet one - the
        400 body carries no "features" key, so code that reaches for it with a
        default reports "no active alerts" for a state that has thirty.
        """
        payload = self._get(
            f"{self.base_url}/alerts/active", params={"area": location.state}
        )
        return payload.get("features") or []

    def fetch_forecast(self, location: ResolvedLocation) -> dict:
        return self._get(
            f"{self.base_url}/gridpoints/{location.office}"
            f"/{location.grid_x},{location.grid_y}/forecast"
        )

    # -- Normalization ------------------------------------------------------

    def alert_documents(
        self, location: ResolvedLocation, features: Iterable[dict]
    ) -> list[WeatherDocument]:
        """Normalize /alerts/active features into WeatherDocuments."""
        documents: list[WeatherDocument] = []

        for feature in features or []:
            properties = feature.get("properties") or {}

            description = (properties.get("description") or "").strip()
            instruction = (properties.get("instruction") or "").strip()
            # Joined because they answer different questions - what is
            # happening, and what to do about it - and a query like "should I
            # evacuate" should be able to match either.
            narrative = "\n\n".join(part for part in (description, instruction) if part)
            if not narrative:
                # Nothing to embed. A vector of empty text carries no signal
                # but would still occupy a slot in every result list.
                continue

            alert_id = properties.get("id") or feature.get("id")
            if not alert_id:
                continue

            issued_at = properties.get("sent")
            documents.append(
                WeatherDocument(
                    id=str(alert_id),
                    # areaDesc names the actual counties under the alert, which
                    # is more precise than the gridpoint we resolved through.
                    location=(properties.get("areaDesc") or location.label),
                    latitude=location.latitude,
                    longitude=location.longitude,
                    source_type=SOURCE_ALERT,
                    event=properties.get("event"),
                    headline=properties.get("headline") or properties.get("event"),
                    narrative_text=narrative,
                    issued_at=issued_at,
                    effective_at=properties.get("effective") or properties.get("onset") or issued_at,
                    expires_at=properties.get("expires") or properties.get("ends"),
                    severity=properties.get("severity"),
                    payload=feature,
                )
            )

        return documents

    def forecast_documents(
        self, location: ResolvedLocation, forecast: dict
    ) -> list[WeatherDocument]:
        """Normalize a gridpoint forecast's periods into WeatherDocuments."""
        properties = (forecast or {}).get("properties") or {}
        generated_at = properties.get("generatedAt") or properties.get("updateTime")

        documents: list[WeatherDocument] = []
        for period in properties.get("periods") or []:
            narrative = (period.get("detailedForecast") or "").strip()
            if not narrative:
                continue

            start, end = period.get("startTime"), period.get("endTime")
            # Keyed on the time window, never the period name. NWS relabels the
            # same window as the day advances ("This Afternoon" is part of
            # "Today" in an earlier issuance), so a name-keyed id would store
            # the same forecast twice under two different keys.
            digest = hashlib.sha256(
                f"{location.label}|{start}|{end}".encode("utf-8")
            ).hexdigest()[:16]

            documents.append(
                WeatherDocument(
                    id=f"forecast:{location.office}/{location.grid_x},{location.grid_y}:{digest}",
                    location=location.label,
                    latitude=location.latitude,
                    longitude=location.longitude,
                    source_type=SOURCE_FORECAST,
                    event=period.get("name"),
                    headline=period.get("shortForecast"),
                    narrative_text=narrative,
                    issued_at=generated_at,
                    effective_at=start,
                    expires_at=end,
                    severity=None,  # forecasts carry no severity classification
                    payload=period,
                )
            )

        return documents

    # -- The whole harvest --------------------------------------------------

    def harvest(
        self,
        locations: Sequence[str],
        limit: int = 50,
        source_types: Sequence[str] = SOURCE_TYPES,
    ) -> list[WeatherDocument]:
        """Fetch and normalize every requested source for every location.

        One unusable location does not lose the rest of the batch: its failure
        is recorded on self.errors and the harvest continues. Documents are
        deduplicated by id, because neighbouring locations share a forecast
        office and will return the same statewide alerts.
        """
        self.errors = []
        wanted = {str(s).strip().lower() for s in source_types}
        seen: set[str] = set()
        documents: list[WeatherDocument] = []

        def collect(batch: list[WeatherDocument]) -> None:
            for document in batch[:limit] if limit else batch:
                if document.id in seen:
                    continue
                seen.add(document.id)
                documents.append(document)

        for raw_location in locations:
            try:
                location = self.resolve(raw_location)

                if SOURCE_ALERT in wanted:
                    collect(self.alert_documents(location, self.fetch_alerts(location)))

                if SOURCE_FORECAST in wanted:
                    collect(
                        self.forecast_documents(location, self.fetch_forecast(location))
                    )

            except Exception as exc:  # noqa: BLE001 - one bad location, not a bad batch
                message = f"{raw_location}: {exc}"
                logger.warning("Skipping location %s", message)
                self.errors.append(message)
                continue

        return documents


__all__ = [
    "GeocodeError",
    "LocationParseError",
    "NWSClient",
    "ParsedLocation",
    "ResolvedLocation",
    "SOURCE_ALERT",
    "SOURCE_FORECAST",
    "SOURCE_TYPES",
    "WeatherClientError",
    "WeatherDocument",
    "content_hash",
    "parse_location",
]

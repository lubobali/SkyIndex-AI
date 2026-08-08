"""Request validation for the SkyIndex-AI API.

Kept out of the route handlers so the rules are testable on their own and each
one is stated once. Every function either returns a clean value or raises
BadRequest with a message a caller can act on.
"""

from __future__ import annotations

from weather_client import SOURCE_TYPES

# Clamped rather than rejected: a caller asking for 500 results wants "as many
# as you can give me", and failing that request helps nobody. The ceiling
# exists because each result carries its chunk text, so an unbounded top_k is
# an unbounded response body.
MIN_TOP_K = 1
MAX_TOP_K = 20
DEFAULT_TOP_K = 5

MAX_QUERY_LENGTH = 1000
MAX_LOCATIONS = 25
MIN_SYNC_LIMIT = 1
MAX_SYNC_LIMIT = 200
DEFAULT_SYNC_LIMIT = 50


class BadRequest(ValueError):
    """A client error with a message safe to return to the caller."""


def clean_query(value: object) -> str:
    """Validate a search query string."""
    if value is None:
        raise BadRequest("Missing required field: query")
    if not isinstance(value, str):
        raise BadRequest(f"query must be a string, got {type(value).__name__}")

    query = value.strip()
    if not query:
        raise BadRequest("query must not be empty")
    if len(query) > MAX_QUERY_LENGTH:
        raise BadRequest(
            f"query is {len(query)} characters, limit is {MAX_QUERY_LENGTH}"
        )
    return query


def clean_top_k(value: object, default: int = DEFAULT_TOP_K) -> int:
    """Coerce and clamp top_k into MIN_TOP_K..MAX_TOP_K."""
    if value is None or value == "":
        return default

    if isinstance(value, bool):
        # bool is an int subclass; True would silently become top_k=1.
        raise BadRequest("top_k must be a number")
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise BadRequest(f"top_k must be a number, got {value!r}") from None

    return max(MIN_TOP_K, min(MAX_TOP_K, number))


def clean_source_type(value: object) -> str | None:
    """Validate an optional source_type filter."""
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise BadRequest("source_type must be a string")

    source_type = value.strip().lower()
    if source_type in ("", "all", "any"):
        return None
    if source_type not in SOURCE_TYPES:
        raise BadRequest(
            f"Unknown source_type {value!r}. Expected one of: "
            f"{', '.join(SOURCE_TYPES)}, or omit it for both."
        )
    return source_type


def clean_locations(value: object, default: list[str]) -> list[str]:
    """Validate the list of locations to sync."""
    if value is None:
        return list(default)
    if isinstance(value, str):
        # A bare string is a common mistake and unambiguous - treat it as one
        # location rather than iterating it character by character.
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise BadRequest("locations must be a list of strings")

    locations = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if not locations:
        raise BadRequest("locations must contain at least one non-empty string")
    if len(locations) > MAX_LOCATIONS:
        raise BadRequest(
            f"{len(locations)} locations requested, limit is {MAX_LOCATIONS}. "
            "Each one costs at least two upstream API calls."
        )
    return locations


def clean_sync_limit(value: object, default: int = DEFAULT_SYNC_LIMIT) -> int:
    """Coerce and clamp the per-location document cap."""
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        raise BadRequest("limit must be a number")
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise BadRequest(f"limit must be a number, got {value!r}") from None

    return max(MIN_SYNC_LIMIT, min(MAX_SYNC_LIMIT, number))


def clean_source_types(value: object) -> list[str]:
    """Validate which sources to harvest. Defaults to all of them."""
    if value is None or value == "":
        return list(SOURCE_TYPES)
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise BadRequest("source_types must be a list of strings")

    requested = [item.strip().lower() for item in value if isinstance(item, str) and item.strip()]
    unknown = [item for item in requested if item not in SOURCE_TYPES]
    if unknown:
        raise BadRequest(
            f"Unknown source_types: {', '.join(unknown)}. "
            f"Expected any of: {', '.join(SOURCE_TYPES)}"
        )
    return requested or list(SOURCE_TYPES)


__all__ = [
    "BadRequest",
    "DEFAULT_SYNC_LIMIT",
    "DEFAULT_TOP_K",
    "MAX_LOCATIONS",
    "MAX_QUERY_LENGTH",
    "MAX_SYNC_LIMIT",
    "MAX_TOP_K",
    "MIN_TOP_K",
    "clean_locations",
    "clean_query",
    "clean_source_type",
    "clean_source_types",
    "clean_sync_limit",
    "clean_top_k",
]

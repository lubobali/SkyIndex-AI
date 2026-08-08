"""Optional natural-language summary over retrieved weather documents.

Retrieval alone answers "which documents match". A user asking "flash flood
risk this weekend" wants the answer, not a ranked list to read themselves.

This is strictly additive. Every failure path returns None and the caller
serves the ranked results unchanged - a summarizer outage must never take the
search endpoint down with it.

Uses a Databricks Foundation Model serving endpoint, so no additional API key
or provider account is needed inside the workspace.
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

logger = logging.getLogger(__name__)

# Databricks now serves managed models through the Unity AI Gateway, addressed
# by their Unity Catalog name rather than the older per-workspace serving
# endpoint name. The Gateway speaks the OpenAI chat-completions shape.
SUMMARY_ENDPOINT = os.environ.get("SUMMARY_ENDPOINT", "system.ai.claude-haiku-4-5")
SUMMARY_GATEWAY_PATH = "/ai-gateway/mlflow/v1/chat/completions"
SUMMARY_MAX_TOKENS = int(os.environ.get("SUMMARY_MAX_TOKENS", "300"))
SUMMARY_MAX_CONTEXT_CHARS = int(os.environ.get("SUMMARY_MAX_CONTEXT_CHARS", "6000"))
SUMMARY_TIMEOUT = int(os.environ.get("SUMMARY_TIMEOUT", "45"))

SYSTEM_PROMPT = (
    "You summarize official National Weather Service text for a general "
    "audience. Use ONLY the numbered passages provided. Never add hazards, "
    "locations, times, or advice that are not in them. If the passages do not "
    "answer the question, say so plainly. Cite the passages you use as [1], "
    "[2]. Be brief: three sentences at most. Lead with anything dangerous."
)


class SummaryUnavailable(RuntimeError):
    """The summarizer could not be reached or returned nothing usable."""


def build_context(results: Sequence[dict], max_chars: int = SUMMARY_MAX_CONTEXT_CHARS) -> str:
    """Render retrieved chunks as a numbered, citable context block.

    Truncated by total characters rather than by result count: chunk lengths
    vary by an order of magnitude between a one-line forecast and a full
    warning, so a fixed count is a poor proxy for how much context is used.
    """
    lines: list[str] = []
    budget = max_chars

    for position, result in enumerate(results, start=1):
        header = (
            f"[{position}] {result.get('source_type', '?')} | "
            f"{result.get('event') or result.get('headline') or 'Weather'} | "
            f"{result.get('location', 'unknown location')}"
        )
        body = " ".join((result.get("chunk_text") or "").split())
        entry = f"{header}\n{body}"

        if len(entry) > budget:
            entry = entry[: max(0, budget)].rstrip()
            if entry:
                lines.append(entry)
            break

        lines.append(entry)
        budget -= len(entry)

    return "\n\n".join(lines)


def summarize(query: str, results: Sequence[dict], endpoint: str | None = None) -> str | None:
    """Summarize the top results in plain language, or return None.

    Returns None rather than raising on every failure path. The summary is a
    bonus on top of retrieval; if it cannot be produced, the caller still has
    ranked results worth serving.
    """
    if not results:
        return None

    # `endpoint or SUMMARY_ENDPOINT` would be wrong: it treats an explicit ""
    # as "use the default", so a caller could not turn summaries off per call.
    # None means "unspecified, use the default"; "" means "disabled".
    target = SUMMARY_ENDPOINT if endpoint is None else endpoint
    if not target:
        return None

    context = build_context(results)
    if not context.strip():
        return None

    try:
        host, token = _credentials()
        if not host or not token:
            logger.info("No Databricks credentials available; skipping summary")
            return None

        import requests

        response = requests.post(
            f"{host.rstrip('/')}{SUMMARY_GATEWAY_PATH}",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={
                "model": target,
                "max_tokens": SUMMARY_MAX_TOKENS,
                # Summarizing retrieved facts, not composing prose. Low
                # temperature because every sentence should be traceable to a
                # passage, and creativity here reads as invented weather.
                "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Question: {query}\n\nPassages:\n{context}\n\n"
                            "Answer the question using only these passages."
                        ),
                    },
                ],
            },
            timeout=SUMMARY_TIMEOUT,
        )
        response.raise_for_status()
        return _extract_text(response.json())

    except Exception:
        # Model not available on this tier, no credentials, network blocked,
        # gateway error - all the same to the caller, which still has results.
        logger.warning("Summary endpoint %s unavailable", target, exc_info=True)
        return None


def _credentials() -> tuple[str | None, str | None]:
    """Resolve the workspace host and a bearer token.

    A deployed Databricks App is issued OAuth credentials for its service
    principal, which the SDK picks up with no configuration. Locally there is
    usually nothing - Free Edition blocks personal access token creation - so
    this returns (None, None) and the caller skips the summary rather than
    failing the search.
    """
    host = os.environ.get("DATABRICKS_HOST")
    token = os.environ.get("DATABRICKS_TOKEN")
    if host and token:
        return host, token

    try:
        from databricks.sdk import WorkspaceClient

        client = WorkspaceClient()
        headers = client.config.authenticate() or {}
        authorization = headers.get("Authorization", "")
        if authorization.startswith("Bearer "):
            return client.config.host, authorization.removeprefix("Bearer ")
    except Exception:
        logger.debug("Could not resolve Databricks credentials", exc_info=True)

    return None, None


def _extract_text(payload) -> str | None:
    """Pull the assistant message out of a chat-completions response.

    Content arrives either as a plain string or as a list of typed blocks
    (the Anthropic-style shape the Gateway passes through for Claude models).
    Both are handled rather than pinning one and breaking on the other.
    """
    choices = (payload or {}).get("choices") if isinstance(payload, dict) else None
    if not choices:
        return None

    message = choices[0].get("message") or {}
    content = message.get("content")

    if isinstance(content, list):
        content = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") in (None, "text")
        )

    return content.strip() if isinstance(content, str) and content.strip() else None


__all__ = [
    "SUMMARY_ENDPOINT",
    "SUMMARY_GATEWAY_PATH",
    "SummaryUnavailable",
    "build_context",
    "summarize",
]

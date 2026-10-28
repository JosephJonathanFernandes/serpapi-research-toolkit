"""Exception hierarchy for langchain_serpapi_provider.

All errors derive from :class:`SerpApiError`, which itself derives from
``RuntimeError`` so code written against the earlier ``RuntimeError``
behaviour keeps working.
"""

from __future__ import annotations


class SerpApiError(RuntimeError):
    """Base class for all errors raised by this package."""

    #: Whether the retriever should retry the request with backoff.
    retryable: bool = False


class SerpApiAuthError(SerpApiError):
    """The API key is missing, invalid or lacks permission. Never retried."""


class SerpApiRateLimitError(SerpApiError):
    """Rate limit hit (too many requests per hour/second). Retried with backoff."""

    retryable = True


class SerpApiQuotaError(SerpApiRateLimitError):
    """Account has run out of searches. Retrying cannot help, so it is not retried."""

    retryable = False


class SerpApiNetworkError(SerpApiError):
    """Connection failure, timeout or 5xx from SerpApi. Retried with backoff."""

    retryable = True


class SerpApiRequestError(SerpApiError):
    """Any other error reported by SerpApi (bad parameters, etc.). Not retried."""


def classify_error_message(message: str) -> type[SerpApiError]:
    """Map a SerpApi ``error`` string onto an exception class."""
    msg = message.lower()
    if "invalid api key" in msg or "unauthorized" in msg or ("api key" in msg and "missing" in msg):
        return SerpApiAuthError
    if "run out of searches" in msg or "out of searches" in msg or "plan" in msg and "limit" in msg:
        return SerpApiQuotaError
    if "rate limit" in msg or "too many requests" in msg or "hourly" in msg:
        return SerpApiRateLimitError
    return SerpApiRequestError


def is_empty_results_message(message: str) -> bool:
    """True for SerpApi's 'no results' error strings, which are not failures."""
    msg = message.lower()
    return "hasn't returned any results" in msg or "hasn’t returned any results" in msg or "no results" in msg

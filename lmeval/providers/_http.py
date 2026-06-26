"""Shared HTTP helper: POST with retry and exponential backoff.

Hosted providers reach across the network, so a single transient failure -- a
rate limit, a 5xx, a dropped connection -- shouldn't fail an eval task outright.
This retries those cases with exponential backoff while letting deterministic
client errors (a bad request, a missing key) fail fast.
"""

import time

import requests

# Status codes worth retrying: rate limiting and transient server errors.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504, 529})


def _retry_after_seconds(resp):
    """Seconds to wait per a numeric Retry-After header, or None if absent."""
    value = resp.headers.get("Retry-After", "")
    return float(value) if value.strip().isdigit() else None


def post_with_retries(url, *, max_retries=3, base_delay=0.5, max_delay=8.0,
                      sleep=time.sleep, **kwargs):
    """POST `url`, retrying transient failures with exponential backoff.

    Retries on connection errors, timeouts, and the status codes in
    RETRY_STATUSES, up to `max_retries` times. A numeric Retry-After header
    overrides the computed backoff for that attempt. A 2xx response is returned;
    a non-retryable 4xx raises immediately. If every attempt fails, the last
    error is raised.
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        delay = None
        try:
            resp = requests.post(url, **kwargs)
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
        else:
            if resp.status_code not in RETRY_STATUSES:
                resp.raise_for_status()  # 2xx returns; non-retryable 4xx raises
                return resp
            last_exc = requests.HTTPError(
                f"{resp.status_code} {resp.reason}", response=resp)
            delay = _retry_after_seconds(resp)

        if attempt >= max_retries:
            break
        if delay is None:
            delay = min(max_delay, base_delay * (2 ** attempt))
        sleep(delay)

    raise last_exc

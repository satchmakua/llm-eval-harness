import pytest
import requests

from lmeval.providers import _http
from lmeval.providers._http import post_with_retries


class FakeResp:
    def __init__(self, status_code=200, headers=None, json_data=None, reason="OK"):
        self.status_code = status_code
        self.headers = headers or {}
        self.reason = reason
        self._json = json_data or {}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} {self.reason}", response=self)


def _patch_post(monkeypatch, items):
    """Make requests.post yield each entry of `items` (a FakeResp or an Exception)."""
    calls = {"n": 0}
    seq = list(items)

    def fake_post(url, **kwargs):
        calls["n"] += 1
        item = seq[min(calls["n"] - 1, len(seq) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(_http.requests, "post", fake_post)
    return calls


def test_returns_on_first_success(monkeypatch):
    calls = _patch_post(monkeypatch, [FakeResp(200, json_data={"ok": 1})])
    resp = post_with_retries("http://x", sleep=lambda *_: None)
    assert resp.json() == {"ok": 1}
    assert calls["n"] == 1


def test_retries_then_succeeds(monkeypatch):
    calls = _patch_post(monkeypatch, [FakeResp(503, reason="Service Unavailable"),
                                      FakeResp(200, json_data={"ok": 1})])
    resp = post_with_retries("http://x", sleep=lambda *_: None)
    assert resp.status_code == 200
    assert calls["n"] == 2


def test_exhausts_retries_and_raises(monkeypatch):
    calls = _patch_post(monkeypatch, [FakeResp(500, reason="err")])
    with pytest.raises(requests.HTTPError):
        post_with_retries("http://x", max_retries=2, sleep=lambda *_: None)
    assert calls["n"] == 3  # initial attempt + 2 retries


def test_does_not_retry_client_error(monkeypatch):
    calls = _patch_post(monkeypatch, [FakeResp(400, reason="Bad Request")])
    with pytest.raises(requests.HTTPError):
        post_with_retries("http://x", sleep=lambda *_: None)
    assert calls["n"] == 1  # 4xx is not retried


def test_retries_connection_error(monkeypatch):
    calls = _patch_post(monkeypatch, [requests.ConnectionError("boom"), FakeResp(200)])
    resp = post_with_retries("http://x", sleep=lambda *_: None)
    assert resp.status_code == 200
    assert calls["n"] == 2


def test_honors_retry_after_header(monkeypatch):
    _patch_post(monkeypatch, [FakeResp(429, headers={"Retry-After": "2"}), FakeResp(200)])
    slept = []
    post_with_retries("http://x", sleep=slept.append)
    assert slept == [2.0]

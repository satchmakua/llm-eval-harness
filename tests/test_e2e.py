"""End-to-end: a real provider adapter talking to a local stub HTTP server.

Unlike the other tests (which stub the provider object), these exercise the
actual `requests` call path -- socket, headers, JSON body, retry/backoff, and
response parsing -- against a server we control.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from lmeval import runner as runner_mod
from lmeval.providers.bedrock import BedrockProvider, sigv4_headers
from lmeval.providers.gemini import GeminiProvider
from lmeval.providers.openai import OpenAIProvider
from lmeval.runner import run_suites
from lmeval.types import Suite, Task


def _openai_body(content, prompt_tokens=11, completion_tokens=3):
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


class StubServer:
    """Replays a queued sequence of (status, json) responses over real HTTP.

    The last queued response repeats once the queue is drained, so a one-entry
    queue answers every request identically. Captures each request for asserts.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self._lock = threading.Lock()
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length else b""
                with outer._lock:
                    outer.requests.append({
                        "path": self.path,
                        "headers": dict(self.headers),
                        "body": json.loads(raw) if raw else None,
                    })
                    idx = min(len(outer.requests) - 1, len(outer._responses) - 1)
                    status, payload = outer._responses[idx]
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass  # keep test output quiet

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)

    @property
    def base_url(self):
        host, port = self._httpd.server_address
        return f"http://{host}:{port}/v1"

    def __enter__(self):
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._httpd.shutdown()
        self._httpd.server_close()


def test_openai_adapter_end_to_end(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with StubServer([(200, _openai_body("positive"))]) as stub:
        provider = OpenAIProvider(base_url=stub.base_url)
        monkeypatch.setattr(runner_mod, "get_provider", lambda name, **kw: provider)
        task = Task(id="t", prompt="classify: great!", system="be terse",
                    graders=[{"type": "contains", "any_of": ["positive"]}])
        suite = Suite(name="s", tasks=[task], models=["openai:gpt-4o-mini"])
        results = run_suites([suite], {"default_provider": "ollama", "model_options": {}})

    assert len(results) == 1
    r = results[0]
    assert r.output == "positive"
    assert r.verdict is True
    assert (r.prompt_tokens, r.completion_tokens) == (11, 3)
    assert r.cost_usd > 0  # priced from PRICING for openai:gpt-4o-mini

    sent = stub.requests[0]
    assert sent["path"] == "/v1/chat/completions"
    assert sent["headers"]["Authorization"] == "Bearer test-key"
    assert sent["body"]["model"] == "gpt-4o-mini"
    assert sent["body"]["messages"][0] == {"role": "system", "content": "be terse"}


def test_openai_adapter_retries_over_socket(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    # first response is a retryable 503, second is a normal 200
    with StubServer([(503, {"error": "busy"}), (200, _openai_body("ok"))]) as stub:
        provider = OpenAIProvider(base_url=stub.base_url)
        comp = provider.complete("gpt-4o-mini", [{"role": "user", "content": "hi"}])

    assert comp.text == "ok"
    assert len(stub.requests) == 2  # adapter retried after the 503


def _gemini_body(content, prompt_tokens=7, completion_tokens=2):
    return {
        "candidates": [{"content": {"role": "model", "parts": [{"text": content}]},
                        "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": prompt_tokens,
                          "candidatesTokenCount": completion_tokens},
    }


def test_gemini_adapter_end_to_end(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    with StubServer([(200, _gemini_body("positive"))]) as stub:
        provider = GeminiProvider(base_url=stub.base_url)
        monkeypatch.setattr(runner_mod, "get_provider", lambda name, **kw: provider)
        task = Task(id="t", prompt="classify: great!", system="be terse",
                    graders=[{"type": "contains", "any_of": ["positive"]}])
        suite = Suite(name="s", tasks=[task], models=["gemini:gemini-2.5-flash"])
        results = run_suites([suite], {"default_provider": "ollama", "model_options": {}})

    r = results[0]
    assert r.output == "positive"
    assert r.verdict is True
    assert (r.prompt_tokens, r.completion_tokens) == (7, 2)
    assert r.cost_usd > 0  # priced from PRICING for gemini:gemini-2.5-flash

    sent = stub.requests[0]
    assert "gemini-2.5-flash:generateContent" in sent["path"]
    assert sent["headers"]["x-goog-api-key"] == "g-key"
    # system prompt is hoisted to systemInstruction; the user role maps through
    assert sent["body"]["systemInstruction"]["parts"][0]["text"] == "be terse"
    assert sent["body"]["contents"][0]["role"] == "user"


def test_gemini_adapter_handles_empty_candidates(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    with StubServer([(200, {"candidates": []})]) as stub:
        provider = GeminiProvider(base_url=stub.base_url)
        comp = provider.complete("gemini-2.5-flash", [{"role": "user", "content": "hi"}])
    assert comp.text == ""           # safety block / empty response -> empty text
    assert comp.prompt_tokens == 0


def test_sigv4_matches_aws_get_vanilla_vector():
    # Official AWS SigV4 test-suite "get-vanilla" known answer (verified against
    # github.com/saibotsivad/aws-sig-v4-test-suite). If this drifts, the signer
    # is wrong -- do not "fix" the expected value.
    headers = sigv4_headers(
        "GET", "https://example.amazonaws.com/",
        region="us-east-1", service="service",
        access_key="AKIDEXAMPLE",
        secret_key="wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
        body=b"", amz_date="20150830T123600Z",
    )
    assert headers["Authorization"] == (
        "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, "
        "SignedHeaders=host;x-amz-date, "
        "Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31"
    )
    assert headers["X-Amz-Date"] == "20150830T123600Z"


def test_sigv4_includes_session_token_when_present():
    headers = sigv4_headers(
        "POST", "https://bedrock-runtime.us-east-1.amazonaws.com/model/m/invoke",
        region="us-east-1", access_key="AKID", secret_key="secret",
        body=b"{}", amz_date="20150830T123600Z", session_token="tok123",
    )
    assert headers["X-Amz-Security-Token"] == "tok123"


def _bedrock_body(content, input_tokens=9, output_tokens=3):
    return {"content": [{"type": "text", "text": content}],
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}}


def test_bedrock_adapter_end_to_end(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    with StubServer([(200, _bedrock_body("positive"))]) as stub:
        provider = BedrockProvider(base_url=stub.base_url)
        monkeypatch.setattr(runner_mod, "get_provider", lambda name, **kw: provider)
        task = Task(id="t", prompt="classify: great!", system="be terse",
                    graders=[{"type": "contains", "any_of": ["positive"]}])
        suite = Suite(name="s", tasks=[task],
                      models=["bedrock:anthropic.claude-haiku-4-5"])
        results = run_suites([suite], {"default_provider": "ollama", "model_options": {}})

    r = results[0]
    assert r.output == "positive"
    assert r.verdict is True
    assert (r.prompt_tokens, r.completion_tokens) == (9, 3)
    assert r.cost_usd > 0  # priced from PRICING

    sent = stub.requests[0]
    assert "/model/anthropic.claude-haiku-4-5/invoke" in sent["path"]
    assert sent["headers"]["Authorization"].startswith("AWS4-HMAC-SHA256 ")
    assert "X-Amz-Date" in sent["headers"]
    # Anthropic-on-Bedrock body shape: version + hoisted system + messages
    assert sent["body"]["anthropic_version"] == "bedrock-2023-05-31"
    assert sent["body"]["system"] == "be terse"
    assert sent["body"]["messages"][0]["role"] == "user"

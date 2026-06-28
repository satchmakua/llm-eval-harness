"""Hosted provider for Anthropic models on Amazon Bedrock.

Calls the Bedrock Runtime `invoke` endpoint with the Anthropic Messages format,
signing requests with AWS Signature V4 (hand-rolled -- no boto3) so the adapter
stays on the same raw-HTTP path as the others.

Credentials come from the standard AWS environment variables: AWS_ACCESS_KEY_ID,
AWS_SECRET_ACCESS_KEY, optional AWS_SESSION_TOKEN, and a region from AWS_REGION /
AWS_DEFAULT_REGION (default us-east-1).
"""

import hashlib
import hmac
import json
import os
import time
from datetime import datetime, timezone
from urllib.parse import quote, urlparse

from ..types import Completion
from ._http import post_with_retries
from .base import Provider

_SERVICE = "bedrock"


def _sign(key, msg):
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def sigv4_headers(method, url, region, access_key, secret_key, body=b"",
                  service=_SERVICE, amz_date=None, session_token=None,
                  extra_headers=None):
    """Return SigV4 auth headers (Authorization, X-Amz-Date, ...) for a request.

    `amz_date` (YYYYMMDDTHHMMSSZ) is injectable for testing; it defaults to now
    (UTC). The URL's path must already be canonically encoded (the caller does
    this for the model id), and query strings are assumed empty -- which holds
    for the Bedrock invoke endpoint.
    """
    parsed = urlparse(url)
    host = parsed.netloc
    canonical_uri = parsed.path or "/"
    canonical_qs = parsed.query
    if amz_date is None:
        amz_date = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    date = amz_date[:8]
    payload_hash = hashlib.sha256(body).hexdigest()

    headers = {"host": host, "x-amz-date": amz_date}
    for k, v in (extra_headers or {}).items():
        headers[k.lower()] = v
    signed_headers = ";".join(sorted(headers))
    canonical_headers = "".join(f"{k}:{headers[k].strip()}\n" for k in sorted(headers))

    canonical_request = "\n".join(
        [method, canonical_uri, canonical_qs, canonical_headers,
         signed_headers, payload_hash])
    scope = f"{date}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join(
        ["AWS4-HMAC-SHA256", amz_date, scope,
         hashlib.sha256(canonical_request.encode()).hexdigest()])

    k_date = _sign(("AWS4" + secret_key).encode(), date)
    k_region = _sign(k_date, region)
    k_service = _sign(k_region, service)
    k_signing = _sign(k_service, "aws4_request")
    signature = hmac.new(k_signing, string_to_sign.encode(), hashlib.sha256).hexdigest()

    auth = (f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}")
    out = {"Authorization": auth, "X-Amz-Date": amz_date}
    if session_token:
        out["X-Amz-Security-Token"] = session_token
    return out


class BedrockProvider(Provider):
    name = "bedrock"

    def __init__(self, base_url=None, region=None, timeout=120, max_retries=3,
                 version="bedrock-2023-05-31"):
        self.region = (region or os.environ.get("AWS_REGION")
                       or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1")
        self.base_url = (base_url
                         or f"https://bedrock-runtime.{self.region}.amazonaws.com").rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.version = version
        self.access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        self.secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        self.session_token = os.environ.get("AWS_SESSION_TOKEN")

    def available(self):
        return bool(self.access_key and self.secret_key)

    def complete(self, model, messages, options=None):
        if not (self.access_key and self.secret_key):
            raise RuntimeError("AWS credentials are not set "
                               "(AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)")
        # Bedrock takes the Anthropic Messages format minus `model`; the system
        # prompt is a separate top-level field.
        system = None
        conversation = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                conversation.append({"role": m["role"], "content": m["content"]})
        body = {
            "anthropic_version": self.version,
            "messages": conversation,
            "max_tokens": (options or {}).get("max_tokens", 1024),
        }
        if system:
            body["system"] = system
        if options and "temperature" in options:
            body["temperature"] = options["temperature"]
        # Serialize ONCE and sign/send the exact same bytes -- re-serializing
        # would change the payload hash and invalidate the signature.
        raw = json.dumps(body).encode()

        # Encode the model id in the path so the wire path matches what we sign
        # (Bedrock ids can contain ':', e.g. inference-profile/versioned ids).
        url = f"{self.base_url}/model/{quote(model, safe='')}/invoke"
        headers = {"content-type": "application/json"}
        headers.update(sigv4_headers(
            "POST", url, self.region, self.access_key, self.secret_key,
            body=raw, session_token=self.session_token,
            extra_headers={"content-type": "application/json"}))

        started = time.time()
        resp = post_with_retries(url, headers=headers, data=raw,
                                 timeout=self.timeout, max_retries=self.max_retries)
        data = resp.json()
        text = "".join(b.get("text", "") for b in data.get("content", [])
                       if b.get("type") == "text")
        usage = data.get("usage", {})
        return Completion(
            text=text,
            model=model,
            provider=self.name,
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            latency_s=round(time.time() - started, 3),
        )

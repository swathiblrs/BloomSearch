"""Provider-neutral client for OpenAI-compatible chat completion APIs.

No model is loaded or executed locally. The API may point to any cloud provider
or hosted LoRA/QLoRA adapter that implements the compatible endpoint.
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ModelAPIError(RuntimeError):
    """Raised when the configured hosted model cannot return a usable answer."""


class HostedModelClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 60,
    ) -> None:
        self.base_url = (base_url or os.getenv("BLOOMSEARCH_API_BASE", "")).rstrip("/")
        self.api_key = api_key or os.getenv("BLOOMSEARCH_API_KEY", "")
        self.model = model or os.getenv("BLOOMSEARCH_MODEL", "")
        self.timeout = timeout
        missing = [
            name
            for name, value in (
                ("BLOOMSEARCH_API_BASE", self.base_url),
                ("BLOOMSEARCH_API_KEY", self.api_key),
                ("BLOOMSEARCH_MODEL", self.model),
            )
            if not value
        ]
        if missing:
            raise ValueError("missing API configuration: " + ", ".join(missing))

    def chat_json(self, *, system: str, user: str, temperature: float = 0) -> Any:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            raise ModelAPIError(f"model API returned HTTP {error.code}: {detail}") from error
        except (URLError, TimeoutError) as error:
            raise ModelAPIError(f"could not reach model API: {error}") from error
        except json.JSONDecodeError as error:
            raise ModelAPIError("model API returned invalid JSON") from error

        try:
            content = body["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise ModelAPIError("model API response did not contain JSON message content") from error

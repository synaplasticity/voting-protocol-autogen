# voting_protocol/anyllm_client.py
"""Custom AnyLLM client for AutoGen.

This client posts to a workspace-scoped AnyLLM /chat endpoint rather than the
OpenAI-style /v1/chat/completions path. It is intended to be referenced by
model_client_cls in the llm_config, and additionally registered at runtime via
agent.client.register_model_client(CustomAnyLLMClient).
"""
from __future__ import annotations

from typing import Any, Dict, List
import httpx

# AutoGen's abstract base model client
from autogen.oai.client import ModelClient


class CustomAnyLLMClient(ModelClient):
    """Custom client to call AnyLLM workspace /chat endpoint.

    Expected config keys in the supplied config dict:
    - model: str
    - base_url: str (must point to the /chat endpoint)
    - api_key: str (workspace API key)
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        self._config = config
        self._url = config.get("base_url")
        self._api_key = config.get("api_key")
        self._model = config.get("model")
        self._timeout = kwargs.get("timeout", 60)

    class _Msg:
        def __init__(self, content: str):
            self.content = content
            self.function_call = None
            self.tool_calls = None

    class _Choice:
        def __init__(self, content: str):
            self.message = CustomAnyLLMClient._Msg(content)

    class _Usage:
        def __init__(self):
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0

    class _Response:
        def __init__(self, text: str, model: str):
            self.choices = [CustomAnyLLMClient._Choice(text)]
            self.model = model
            self.usage = CustomAnyLLMClient._Usage()

    def _join_messages(self, params: Dict[str, Any]) -> str:
        # Convert OpenAI-style messages into a single text string for AnyLLM /chat
        if "messages" in params and isinstance(params["messages"], list):
            parts: List[str] = []
            for m in params["messages"]:
                if isinstance(m, dict):
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    parts.append(f"{role}: {content}")
            return "\n".join(parts)
        return params.get("prompt", "")

    def create(self, params: Dict[str, Any]):
        text = self._join_messages(params)
        payload = {
            "model": self._model,
            "mode": "chat",
            "message": text,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        resp = httpx.post(self._url, json=payload, headers=headers, timeout=self._timeout)
        resp.raise_for_status()
        data: Dict[str, Any] = {}
        if resp.headers.get("content-type", "").startswith("application/json"):
            try:
                data = resp.json()
            except Exception:
                data = {}
        reply = data.get("message") or data.get("response") or data.get("text") or resp.text
        return CustomAnyLLMClient._Response(reply, self._model or "")

    def message_retrieval(self, response: Any):
        # Return list of strings; AutoGen will wrap into message dicts as needed
        return [getattr(c.message, "content", "") for c in response.choices]

    def cost(self, response: Any) -> float:
        return 0.0

    @staticmethod
    def get_usage(response: Any) -> Dict[str, Any]:
        return {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
            "completion_tokens": getattr(response.usage, "completion_tokens", 0),
            "total_tokens": getattr(response.usage, "total_tokens", 0),
            "cost": 0.0,
            "model": getattr(response, "model", ""),
        }
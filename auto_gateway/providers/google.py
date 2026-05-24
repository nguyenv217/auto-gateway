from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from .base import BaseProvider, ProviderCallResult


class GoogleProvider(BaseProvider):
    """Provider for google-genai.

    Notes:
    - `call()` is executed in a thread.
    - `call_stream()` is implemented as a best-effort streaming wrapper around
      non-streaming generation until proper async streaming is available.
    """

    def __init__(
        self,
        name: str = "google",
        keys: list[str] | None = None,
        model_configs: dict[str, list[str]] | None = None,
        key_aliases: dict[str, str] | None = None,
    ):
        super().__init__(name=name, keys=keys, models=model_configs or {}, key_aliases=key_aliases)

        self._file_cache: dict[str, tuple[Any, str]] = {}

    async def call(
        self,
        *,
        key: str | None,
        model: str,
        messages: list[dict[str, Any]],
        timeout: float,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
        extra_body: dict[str, Any] | None = None,
    ) -> ProviderCallResult:
        del tools, tool_choice, extra_body
        if not key:
            raise ValueError("GoogleProvider requires an api key")

        def _sync_call() -> ProviderCallResult:
            from google import genai
            from google.genai import types
            import warnings
            import base64

            client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=timeout * 1000))

            system_instruction = None
            gemini_contents = []

            for msg in messages:
                role = msg.get("role")
                content = msg.get("content")

                if role == "system":
                    system_instruction = content
                elif role == "user":
                    if isinstance(content, str):
                        gemini_contents.append(
                            types.Content(role="user", parts=[types.Part.from_text(text=content)])
                        )
                    elif isinstance(content, list):
                        parts = []
                        for item in content:
                            if item.get("type") == "text" and item.get("text"):
                                parts.append(types.Part.from_text(text=item["text"]))
                            elif item.get("type") == "image_url":
                                url_data = item["image_url"]["url"]
                                mime_type = url_data.split(";")[0].split(":")[1]
                                b64_data = url_data.split(",")[1]
                                raw_bytes = base64.b64decode(b64_data)
                                parts.append(types.Part.from_bytes(data=raw_bytes, mime_type=mime_type))
                        if parts:
                            gemini_contents.append(types.Content(role="user", parts=parts))
                elif role == "assistant":
                    parts = []
                    if content:
                        parts.append(types.Part.from_text(text=content))
                    if msg.get("tool_calls"):
                        for tc in msg["tool_calls"]:
                            func = tc.get("function", {})
                            name = func.get("name")
                            args_str = func.get("arguments", "{}")
                            args_dict = json.loads(args_str) if isinstance(args_str, str) else args_str
                            part = types.Part.from_function_call(name=name, args=args_dict)
                            part.thought_signature = "skip_thought_signature_validator"
                            parts.append(part)
                    if parts:
                        gemini_contents.append(types.Content(role="model", parts=parts))
                elif role == "tool":
                    # minimal tool mapping
                    response_dict = {"result": content}
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, dict):
                            response_dict = parsed
                    except Exception:
                        pass
                    part = types.Part.from_function_response(name=msg.get("name", "tool"), response=response_dict)
                    if gemini_contents and gemini_contents[-1].role == "user":
                        gemini_contents[-1].parts.append(part)
                    else:
                        gemini_contents.append(types.Content(role="user", parts=[part]))

            config_kwargs = {
                "system_instruction": system_instruction,
                "safety_settings": [
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                ],
            }

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                response = client.models.generate_content(model=model, contents=gemini_contents, config=types.GenerateContentConfig(**config_kwargs))

            extracted_text = ""
            extracted_reasoning = ""
            normalized_tool_calls = []

            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if getattr(part, "thought", None):
                        extracted_reasoning = part.text
                    elif getattr(part, "text", None):
                        extracted_text = part.text
                    elif getattr(part, "function_call", None):
                        fc = part.function_call
                        try:
                            args_dict = getattr(fc, "args", {})
                            if not args_dict and hasattr(fc, "to_dict"):
                                args_dict = fc.to_dict().get("args", {})
                            args_str = json.dumps(args_dict)
                        except Exception:
                            args_str = "{}"

                        tc_dict = {
                            "id": f"call_{uuid.uuid4().hex[:10]}",
                            "type": "function",
                            "function": {"name": fc.name, "arguments": args_str},
                        }
                        normalized_tool_calls.append(tc_dict)

            usage = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = {
                    "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                    "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                    "total_tokens": getattr(response.usage_metadata, "total_token_count", 0),
                }

            return {
                "text": extracted_text,
                "reasoning": extracted_reasoning,
                "tool_calls": normalized_tool_calls or None,
                "usage": usage,
            }

        return await asyncio.to_thread(_sync_call)

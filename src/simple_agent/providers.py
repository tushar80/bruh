"""Providers for the opencode-go endpoint.

Two backends are supported, both OpenCode Zen Go:
  - OpenAI completions  -> https://opencode.ai/zen/go/v1/chat/completions
  - Anthropic messages  -> https://opencode.ai/zen/go/v1/messages

Both stream SSE responses with httpx. The agent loop calls
``stream()`` and accumulates text + tool calls from the event stream.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

# --------------------------------------------------------------------------- #
# Message types                                                                #
# --------------------------------------------------------------------------- #


@dataclass
class TextBlock:
    text: str = ""


@dataclass
class ThinkingBlock:
    thinking: str = ""
    signature: str = ""


@dataclass
class ToolUseBlock:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResultBlock:
    tool_use_id: str
    content: str
    is_error: bool = False


ContentBlock = TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock


@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: list[ContentBlock]


# --------------------------------------------------------------------------- #
# Token estimation (chars/4 heuristic, conservative overestimate)            #
# --------------------------------------------------------------------------- #

# Roughly 1 token per 4 chars; overestimates slightly (safer for budget checks).
ESTIMATED_IMAGE_CHARS = 4800


def estimate_tokens(message: Message) -> int:
    """Estimate token count for a Message using the chars/4 heuristic."""
    chars = 0
    for block in message.content:
        if isinstance(block, TextBlock):
            chars += len(block.text)
        elif isinstance(block, ThinkingBlock):
            chars += len(block.thinking)
        elif isinstance(block, ToolUseBlock):
            chars += len(block.name) + len(json.dumps(block.arguments, default=str))
        elif isinstance(block, ToolResultBlock):
            chars += len(block.content)
    return -(-chars // 4)  # ceil


def estimate_context_tokens(messages: list[Message], system: str = "") -> int:
    """Estimate total tokens across system + all messages."""
    total = len(system)
    for msg in messages:
        total += estimate_tokens(msg)
    return -(-total // 4)


# --------------------------------------------------------------------------- #
# Tool / usage / events                                                        #
# --------------------------------------------------------------------------- #


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Event:
    """A single streamed event from the provider."""
    type: str  # "text" | "thinking" | "tool_start" | "tool_delta" | "tool_end" | "done" | "error"
    text: str = ""
    tool_id: str = ""
    tool_name: str = ""
    tool_args_json: str = ""
    stop_reason: str = ""
    usage: Usage | None = None
    error: str = ""


# --------------------------------------------------------------------------- #
# Provider protocol                                                            #
# --------------------------------------------------------------------------- #


class Provider(Protocol):
    name: str

    def stream(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> Iterator[Event]: ...


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #

# Base URL for the OpenCode Go subscription endpoint. Individual endpoints:
#   OpenAI-compatible models  -> {base}/chat/completions
#   Anthropic-format models   -> {base}/messages
# Model -> format mapping (see https://opencode.ai/docs/go):
#   openai-compatible : glm-5.2, glm-5.1, grok-4.5, kimi-k3, kimi-k2.7-code,
#                       kimi-k2.6, deepseek-v4-pro, deepseek-v4-flash,
#                       mimo-v2.5, mimo-v2.5-pro, hy3
#   anthropic         : minimax-m3, minimax-m2.7, minimax-m2.5, qwen3.7-max,
#                       qwen3.7-plus, qwen3.6-plus
DEFAULT_BASE_URL = "https://opencode.ai/zen/go/v1"


def get_config() -> dict[str, str | None]:
    return {
        "api_key": os.environ.get("OPENCODE_API_KEY"),
        "base_url": os.environ.get("OPENCODE_BASE_URL", DEFAULT_BASE_URL),
    }


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _parse_sse_lines(line_iter: Iterator[str]) -> Iterator[dict[str, Any]]:
    """Yield parsed JSON payloads from an SSE byte stream, one per ``data:``
    line. Skips comments, pings, and blank lines."""
    for line in line_iter:
        line = line.strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload == "[DONE]":
                return
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                continue


# --------------------------------------------------------------------------- #
# OpenAI completions provider                                                  #
# --------------------------------------------------------------------------- #


class OpenAICompletionsProvider:
    name = "openai-completions"

    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL, timeout: float = 300):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def stream(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        model: str,
        max_tokens: int = 8192,
        temperature: float = 0.7,
    ) -> Iterator[Event]:
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "messages": self._build_messages(system, messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in tools
            ]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        tool_buffers: dict[int, dict[str, str]] = {}

        with httpx.stream("POST", url, json=payload, headers=headers, timeout=self.timeout) as resp:
            if resp.status_code != 200:
                body = resp.read().decode("utf-8", errors="replace")
                yield Event(type="error", error=f"HTTP {resp.status_code}: {body[:500]}")
                return

            for chunk in _parse_sse_lines(resp.iter_lines()):
                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}

                # Reasoning/reasoning_content field (some providers use "reasoning")
                reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                if reasoning:
                    yield Event(type="thinking", text=reasoning)

                if delta.get("content"):
                    yield Event(type="text", text=delta["content"])

                for tc in (delta.get("tool_calls") or []):
                    idx = tc.get("index", 0)
                    buf = tool_buffers.setdefault(idx, {"id": "", "name": "", "args": ""})
                    if tc.get("id"):
                        buf["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        buf["name"] = fn["name"]
                    if fn.get("arguments"):
                        buf["args"] += fn["arguments"]
                    if buf["id"] and buf["name"] and "name" not in buf.get("emitted", {}):
                        buf["emitted"] = {"name": True}  # type: ignore[assignment]
                        yield Event(type="tool_start", tool_id=buf["id"], tool_name=buf["name"])

                finish = choice.get("finish_reason")
                if finish:
                    for buf in tool_buffers.values():
                        if buf.get("id"):
                            yield Event(
                                type="tool_end",
                                tool_id=buf["id"],
                                tool_name=buf.get("name", ""),
                                tool_args_json=buf.get("args", ""),
                            )
                    usage_raw = chunk.get("usage") or {}
                    usage = Usage(
                        input_tokens=usage_raw.get("prompt_tokens", 0),
                        output_tokens=usage_raw.get("completion_tokens", 0),
                    )
                    stop = finish
                    if stop == "tool_calls":
                        stop = "tool_use"
                    yield Event(type="done", stop_reason=stop, usage=usage)

    def _build_messages(self, system: str, messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if system:
            out.append({"role": "system", "content": system})
        for msg in messages:
            if msg.role == "assistant":
                text_parts: list[str] = []
                tool_calls: list[dict[str, Any]] = []
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)
                    elif isinstance(block, ThinkingBlock):
                        pass  # OpenAI completions API doesn't round-trip thinking
                    elif isinstance(block, ToolUseBlock):
                        tool_calls.append({
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.arguments),
                            },
                        })
                entry: dict[str, Any] = {"role": "assistant"}
                if text_parts:
                    entry["content"] = "\n".join(text_parts)
                if tool_calls:
                    entry["tool_calls"] = tool_calls
                out.append(entry)
            else:  # user
                text_parts2: list[str] = []
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        text_parts2.append(block.text)
                    elif isinstance(block, ToolResultBlock):
                        out.append({
                            "role": "tool",
                            "tool_call_id": block.tool_use_id,
                            "content": block.content,
                        })
                if text_parts2:
                    out.append({"role": "user", "content": "\n".join(text_parts2)})
        return out


# --------------------------------------------------------------------------- #
# Anthropic messages provider                                                  #
# --------------------------------------------------------------------------- #


class AnthropicMessagesProvider:
    name = "anthropic-messages"

    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL, timeout: float = 300):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def stream(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        model: str,
        max_tokens: int = 8192,
        temperature: float = 0.7,
    ) -> Iterator[Event]:
        url = f"{self.base_url}/messages"
        payload: dict[str, Any] = {
            "model": model,
            "messages": self._build_messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools
            ]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        with httpx.stream("POST", url, json=payload, headers=headers, timeout=self.timeout) as resp:
            if resp.status_code != 200:
                body = resp.read().decode("utf-8", errors="replace")
                yield Event(type="error", error=f"HTTP {resp.status_code}: {body[:500]}")
                return

            current_tool_id: str | None = None
            current_tool_json = ""

            for chunk in _parse_sse_lines(resp.iter_lines()):
                etype = chunk.get("type", "")

                if etype == "content_block_start":
                    block = chunk.get("content_block", {})
                    btype = block.get("type")
                    if btype == "tool_use":
                        current_tool_id = block.get("id", "")
                        current_tool_json = ""
                        yield Event(
                            type="tool_start",
                            tool_id=current_tool_id,
                            tool_name=block.get("name", ""),
                        )
                    elif btype == "thinking":
                        yield Event(type="thinking", text="")

                elif etype == "content_block_delta":
                    delta = chunk.get("delta", {})
                    dtype = delta.get("type")
                    if dtype == "text_delta":
                        yield Event(type="text", text=delta.get("text", ""))
                    elif dtype == "thinking_delta":
                        yield Event(type="thinking", text=delta.get("thinking", ""))
                    elif dtype == "input_json_delta":
                        current_tool_json += delta.get("partial_json", "")
                        if current_tool_id:
                            yield Event(
                                type="tool_delta",
                                tool_id=current_tool_id,
                                tool_args_json=current_tool_json,
                            )

                elif etype == "content_block_stop":
                    if current_tool_id:
                        yield Event(
                            type="tool_end",
                            tool_id=current_tool_id,
                            tool_args_json=current_tool_json,
                        )
                        current_tool_id = None
                        current_tool_json = ""

                elif etype == "message_delta":
                    d = chunk.get("delta", {})
                    usage_raw = (chunk.get("usage") or {}).get("output_tokens", 0)
                    stop = d.get("stop_reason", "")
                    if stop == "tool_use":
                        stop = "tool_use"
                    yield Event(
                        type="done",
                        stop_reason=stop,
                        usage=Usage(output_tokens=usage_raw),
                    )

                elif etype == "message_start":
                    msg = chunk.get("message", {})
                    uin = (msg.get("usage") or {}).get("input_tokens", 0)
                    if uin:
                        yield Event(type="done", usage=Usage(input_tokens=uin))

                elif etype == "error":
                    yield Event(type="error", error=json.dumps(chunk.get("error", {})))

    def _build_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for msg in messages:
            blocks: list[dict[str, Any]] = []
            for block in msg.content:
                if isinstance(block, TextBlock):
                    blocks.append({"type": "text", "text": block.text})
                elif isinstance(block, ThinkingBlock):
                    blocks.append({"type": "thinking", "thinking": block.text, "signature": block.signature})
                elif isinstance(block, ToolUseBlock):
                    blocks.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.arguments,
                    })
                elif isinstance(block, ToolResultBlock):
                    blocks.append({
                        "type": "tool_result",
                        "tool_use_id": block.tool_use_id,
                        "content": block.content,
                        "is_error": block.is_error,
                    })
            out.append({"role": msg.role, "content": blocks})
        return out


# --------------------------------------------------------------------------- #
# Factory                                                                      #
# --------------------------------------------------------------------------- #


def create_provider(fmt: str = "openai") -> Provider:
    cfg = get_config()
    api_key = cfg["api_key"] or ""
    base_url = cfg["base_url"] or DEFAULT_BASE_URL
    if not api_key:
        raise RuntimeError("OPENCODE_API_KEY environment variable is not set")
    if fmt == "anthropic":
        return AnthropicMessagesProvider(api_key, base_url)  # type: ignore[return-value]
    return OpenAICompletionsProvider(api_key, base_url)  # type: ignore[return-value]
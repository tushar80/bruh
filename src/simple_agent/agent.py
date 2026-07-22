"""The agent loop: stream completions, execute tools, repeat until done.

Run interactively with:

    uv run simple-agent
    uv run simple-agent -p "do something"
    uv run simple-agent --provider anthropic
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from typing import Any

from .providers import (
    AnthropicMessagesProvider,
    ContentBlock,
    Event,
    Message,
    OpenAICompletionsProvider,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    get_config,
)
from .tools import TOOL_SPECS, run_tool

DEFAULT_MODEL = "opencode-go/glm-5.2"
MAX_STEPS = 50
MAX_TOOL_OUTPUT = 20_000

SYSTEM_PROMPT = """\
You are a helpful coding agent operating in the user's local repository.

You can run shell commands, read, write, and edit files, and search the codebase
using the provided tools. Work step by step:

1. Explore: read files, grep, glob, run bash to understand the task.
2. Plan: briefly outline your approach before editing.
3. Act: make edits with the write/edit tools.
4. Verify: run tests / lint / the relevant command before declaring done.

Be concise in your messages to the user. Prefer one tool call per step when
exploring. When you have finished the task, respond with text (no tool call)
describing what you did.

Working directory is the current directory unless the user says otherwise.
"""


def _build_tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            name=spec["name"],
            description=spec["description"],
            input_schema=spec["input_schema"],
        )
        for spec in TOOL_SPECS
    ]


class Agent:
    def __init__(
        self,
        provider: OpenAICompletionsProvider | AnthropicMessagesProvider,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        max_steps: int = MAX_STEPS,
        verbose: bool = False,
    ):
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_steps = max_steps
        self.verbose = verbose
        self.messages: list[Message] = []
        self.tools = _build_tool_specs()

    # --- public ------------------------------------------------------------- #

    def run(self, user_text: str) -> str:
        """Send one user turn; drive the tool loop to completion."""
        self.messages.append(Message(role="user", content=[TextBlock(text=user_text)]))
        return self._drive_loop()

    # --- loop --------------------------------------------------------------- #

    def _drive_loop(self) -> str:
        final_text_parts: list[str] = []
        for step in range(1, self.max_steps + 1):
            new_blocks, stop, text_out, did_tools = self._stream_step()
            if text_out:
                final_text_parts.append(text_out)
            if not new_blocks:
                break
            self.messages.append(Message(role="assistant", content=new_blocks))

            if not did_tools:
                # assistant finished with a plain message
                if self.verbose:
                    print(f"\n[step {step}] assistant done ({stop})", file=sys.stderr)
                break

            # Execute every tool call from this turn
            tool_results: list[ContentBlock] = []
            for block in new_blocks:
                if isinstance(block, ToolUseBlock):
                    if self.verbose:
                        print(f"\n[step {step}] tool: {block.name}", file=sys.stderr)
                    result_text = run_tool(block.name, block.arguments)
                    if self.verbose:
                        snippet = result_text[:300].replace("\n", "\n  ")
                        print(f"  -> {snippet}{'...' if len(result_text) > 300 else ''}", file=sys.stderr)
                    tool_results.append(
                        ToolResultBlock(
                            tool_use_id=block.id,
                            content=result_text[:MAX_TOOL_OUTPUT],
                            is_error=result_text.startswith("[error"),
                        )
                    )
            if tool_results:
                self.messages.append(Message(role="user", content=tool_results))
        else:
            final_text_parts.append("\n[reached max steps]")

        return "\n".join(final_text_parts)

    def _stream_step(self) -> tuple[list[ContentBlock], str, str, bool]:
        """One provider round-trip with streaming. Returns:
        (assistant_blocks, stop_reason, seen_text, had_tool_calls)."""
        text_buf = ""
        thinking_shown = False
        tool_calls: dict[str, dict[str, Any]] = {}
        stop_reason = ""
        had_error = False
        seen_text_parts: list[str] = []

        stream: Iterator[Event] = self.provider.stream(
            system=SYSTEM_PROMPT,
            messages=self.messages,
            tools=self.tools,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        for ev in stream:
            if ev.type == "thinking":
                if ev.text:
                    if self.verbose and not thinking_shown:
                        print("\n[thinking] ", end="", file=sys.stderr, flush=True)
                        thinking_shown = True
                    if self.verbose:
                        print(ev.text, end="", file=sys.stderr, flush=True)
            elif ev.type == "text":
                if ev.text:
                    print(ev.text, end="", flush=True)
                    text_buf += ev.text
                    seen_text_parts.append(ev.text)
            elif ev.type == "tool_start":
                if self.verbose:
                    print(f"\n[tool: {ev.tool_name} ({ev.tool_id})]", file=sys.stderr)
                tool_calls[ev.tool_id] = {"name": ev.tool_name, "args": ""}
            elif ev.type == "tool_delta":
                if ev.tool_id in tool_calls and ev.tool_args_json:
                    tool_calls[ev.tool_id]["args"] = ev.tool_args_json
            elif ev.type == "tool_end":
                entry = tool_calls.get(ev.tool_id)
                if entry is None:
                    tool_calls[ev.tool_id] = {"name": ev.tool_name, "args": ev.tool_args_json}
                else:
                    entry["args"] = ev.tool_args_json or entry["args"]
            elif ev.type == "done":
                if ev.stop_reason:
                    stop_reason = ev.stop_reason
            elif ev.type == "error":
                print(f"\n[error] {ev.error}", file=sys.stderr)
                had_error = True

        if had_error and not tool_calls and not text_buf:
            return [], stop_reason, "", False

        # Build assistant content blocks for history
        new_blocks: list[ContentBlock] = []
        if text_buf:
            new_blocks.append(TextBlock(text=text_buf))
        for tool_id, entry in tool_calls.items():
            args: dict[str, Any] = {}
            raw = entry.get("args", "")
            if raw:
                try:
                    args = json.loads(raw)
                except json.JSONDecodeError:
                    args = {"_raw": raw, "_error": "invalid JSON arguments"}
            new_blocks.append(ToolUseBlock(id=tool_id, name=entry["name"], arguments=args))

        if text_buf:
            print()  # newline after streamed text

        return new_blocks, stop_reason, text_buf, len(tool_calls) > 0


# --------------------------------------------------------------------------- #
# Interactive session                                                          #
# --------------------------------------------------------------------------- #


def _make_provider(fmt: str) -> OpenAICompletionsProvider | AnthropicMessagesProvider:
    cfg = get_config()
    api_key = cfg["api_key"] or ""
    base_url = cfg["base_url"] or "https://opencode.ai/zen/go/v1"
    if not api_key:
        print("error: set OPENCODE_API_KEY before running", file=sys.stderr)
        sys.exit(2)
    if fmt == "anthropic":
        return AnthropicMessagesProvider(api_key, base_url)
    return OpenAICompletionsProvider(api_key, base_url)


def main() -> None:
    p = argparse.ArgumentParser(
        prog="simple-agent",
        description="A pi-style coding agent powered by opencode-go (glm-5.2)",
    )
    p.add_argument("-p", "--prompt", help="Run a single prompt and exit (non-interactive)")
    p.add_argument("--provider", choices=["openai", "anthropic"], default="openai",
                   help="API format to use against opencode-go")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Model id (default: {DEFAULT_MODEL})")
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-steps", type=int, default=MAX_STEPS, help="Max tool-call steps per turn")
    p.add_argument("-v", "--verbose", action="store_true", help="Print tool calls + thinking to stderr")
    args = p.parse_args()

    provider = _make_provider(args.provider)
    agent = Agent(
        provider=provider,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_steps=args.max_steps,
        verbose=args.verbose,
    )

    if args.prompt:
        agent.run(args.prompt)
        return

    print("simple-agent ready. ^D to exit.\n", file=sys.stderr)
    while True:
        try:
            line = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print("\nbye.", file=sys.stderr)
            return
        if not line.strip():
            continue
        agent.run(line)
        print(file=sys.stderr)


if __name__ == "__main__":
    main()
"""The agent loop: stream completions, execute tools, repeat until done.

Run interactively with:

    uv run bruh
    uv run bruh -p "do something"
    uv run bruh --provider anthropic
"""

from __future__ import annotations

import argparse
import json
import os
import readline  # imported for its side effect: line editing and history in input()
import select
import shlex
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from . import render
from .prompts import build_system_prompt
from .providers import (
    DEFAULT_BASE_URL,
    AnthropicMessagesProvider,
    ContentBlock,
    Event,
    Message,
    OpenAICompletionsProvider,
    Provider,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
    estimate_context_tokens,
    estimate_tokens,
    merge_usage,
)
from .tools import TOOL_SPECS, kill_running, run_tool

DEFAULT_MODEL = "deepseek-v4.1-flash"
MAX_STEPS = 50
TOOL_WORKERS = 8
PASTE_WAIT = 0.05


class Agent:
    def __init__(
        self,
        provider: Provider,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 8192,
        temperature: float | None = None,
        max_steps: int = MAX_STEPS,
        verbose: bool = False,
        include_rules: bool = False,
        include_skills: bool = True,
    ):
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_steps = max_steps
        self.out = render.Renderer(verbose=verbose)
        self.messages: list[Message] = []
        self.system_prompt = build_system_prompt(include_rules=include_rules, include_skills=include_skills)
        self.tools = [ToolSpec(**spec) for spec in TOOL_SPECS]
        self.last_usage: Usage | None = None
        self.last_usage_index: int = -1

    def run(self, user_text: str) -> str:
        self.messages.append(Message(role="user", content=[TextBlock(text=user_text)]))
        return self._drive_loop()

    def _context_tokens(self) -> int:
        """Context estimate anchored to the last real provider usage when
        available; falls back to the chars/4 heuristic for everything."""
        if self.last_usage is None or self.last_usage_index < 0:
            return estimate_context_tokens(self.messages, self.system_prompt)
        trailing = sum(
            estimate_tokens(msg) for msg in self.messages[self.last_usage_index + 1:]
        )
        return self.last_usage.total_tokens + trailing

    def _execute_tool_turn(self, new_blocks: list[ContentBlock]) -> list[ContentBlock]:
        """Results come back in wire order: the messages API requires
        tool_result ids to line up with the prior tool_use ids."""
        calls: list[ToolUseBlock] = [
            b for b in new_blocks if isinstance(b, ToolUseBlock)
        ]
        if not calls:
            return []
        # With one call the header goes out before the tool runs, so a slow
        # command is visible while it works. A batch has to buffer each block
        # until it completes, or the headers and bodies interleave.
        single = len(calls) == 1
        if single:
            self.out.tool_header(calls[0].name, calls[0].arguments)
        else:
            self.out.parallel_note(len(calls))

        def _run_one(call: ToolUseBlock) -> str:
            started = time.monotonic()
            result = run_tool(call.name, call.arguments)
            self.out.tool_result(
                call.name,
                call.arguments,
                result,
                time.monotonic() - started,
                header_done=single,
            )
            return result

        pool = ThreadPoolExecutor(max_workers=TOOL_WORKERS)
        try:
            if single:
                ordered: list[str] = [_run_one(calls[0])]
            else:
                ordered = list(pool.map(_run_one, calls))
        except KeyboardInterrupt:
            kill_running()
            pool.shutdown(cancel_futures=True)
            # Substitute error results so every tool_use still gets a
            # tool_result; the messages API rejects dangling tool calls.
            self.out.warn("aborted")
            ordered = ["[error: aborted]"] * len(calls)
        finally:
            pool.shutdown()

        return [
            ToolResultBlock(
                tool_use_id=call.id,
                content=text,
                is_error=text.startswith("[error"),
            )
            for call, text in zip(calls, ordered)
        ]

    def _drive_loop(self) -> str:
        final_text_parts: list[str] = []
        for step in range(1, self.max_steps + 1):
            new_blocks, stop, text_out, did_tools, turn_usage = self._stream_step()
            if text_out:
                final_text_parts.append(text_out)
            if not new_blocks:
                break
            self.messages.append(Message(role="assistant", content=new_blocks))
            if turn_usage is not None:
                self.last_usage = turn_usage
                self.last_usage_index = len(self.messages) - 1

            if not did_tools:
                self.out.debug(f"step {step} done ({stop})")
                break

            self.out.tokens(
                self._context_tokens(),
                self.last_usage.total_tokens if self.last_usage is not None else None,
            )

            tool_results = self._execute_tool_turn(new_blocks)
            if tool_results:
                self.messages.append(Message(role="user", content=tool_results))
        else:
            self.out.warn("reached max steps")
            final_text_parts.append("\n[reached max steps]")

        return "\n".join(final_text_parts)

    def _stream_step(self) -> tuple[list[ContentBlock], str, str, bool, Usage | None]:
        """One provider round-trip with streaming. Returns:
        (assistant_blocks, stop_reason, seen_text, had_tool_calls, turn_usage)."""
        text_buf = ""
        thinking_buf = ""
        thinking_sig = ""
        tool_calls: dict[str, dict[str, Any]] = {}
        stop_reason = ""
        had_error = False
        turn_usage: Usage | None = None

        stream: Iterator[Event] = self.provider.stream(
            system=self.system_prompt,
            messages=self.messages,
            tools=self.tools,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        try:
            for ev in stream:
                if ev.type == "thinking":
                    self.out.thinking(ev.text)
                    thinking_buf += ev.text
                    thinking_sig = ev.signature or thinking_sig
                elif ev.type == "text":
                    if ev.text:
                        self.out.text(ev.text)
                        text_buf += ev.text
                elif ev.type == "tool_start":
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
                    if ev.usage is not None:
                        turn_usage = merge_usage(turn_usage, ev.usage)
                elif ev.type == "error":
                    self.out.error(ev.error)
                    had_error = True
        except KeyboardInterrupt:
            # Ctrl+C during streaming: abort the turn but keep the session alive.
            # httpx.stream's context manager closes the underlying request on the
            # way out. Drop in-flight tool calls so we don't leave the assistant
            # message with a tool_use that has no matching tool_result.
            self.out.warn("aborted")
            stop_reason = "aborted"
            tool_calls = {}

        self.out.end_text()

        if had_error and not tool_calls and not text_buf:
            return [], stop_reason, "", False, turn_usage

        # Thinking goes first: the messages API requires it back (with its
        # signature) on any turn that follows a thinking-enabled response.
        new_blocks: list[ContentBlock] = []
        if thinking_buf:
            new_blocks.append(ThinkingBlock(thinking=thinking_buf, signature=thinking_sig))
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

        return new_blocks, stop_reason, text_buf, len(tool_calls) > 0, turn_usage


def _make_provider(fmt: str) -> Provider:
    api_key = os.environ.get("BRUH_API_KEY", "")
    base_url = os.environ.get("BRUH_BASE_URL") or DEFAULT_BASE_URL
    if not api_key:
        print(render.fg("error", "error: set BRUH_API_KEY before running"), file=sys.stderr)
        sys.exit(2)
    if fmt == "anthropic":
        return AnthropicMessagesProvider(api_key, base_url)
    return OpenAICompletionsProvider(api_key, base_url)


def _edit(out: render.Renderer, draft: str) -> str:
    """Open $VISUAL/$EDITOR on draft. A nonzero exit (vim's :cq) cancels."""
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
    with tempfile.NamedTemporaryFile("w+", suffix=".md", prefix="bruh-", encoding="utf-8") as f:
        f.write(draft)
        f.flush()
        # shell=True because EDITOR may carry arguments, e.g. "code --wait".
        code = subprocess.run(f"{editor} {shlex.quote(f.name)}", shell=True).returncode
        if code != 0:
            out.warn(f"{editor} exited {code}, message not sent")
            return ""
        # By path, not f.read(): editors that save via rename leave f on the old file.
        with open(f.name, encoding="utf-8") as saved:
            return saved.read().strip()


def _read_message(out: render.Renderer) -> tuple[list[str], str]:
    """Returns the lines as typed (for the renderer to erase) and the message.
    Pasted text arrives as one burst, so input still waiting after Enter is the
    rest of a paste; it opens in the editor rather than sending unseen."""
    typed = [input(out.prompt())]
    if not sys.stdin.isatty():
        return typed, typed[0]
    while select.select([sys.stdin], [], [], PASTE_WAIT)[0]:
        typed.append(input(out.continuation()))
    if typed == ["/edit"]:
        return typed, _edit(out, "")
    if len(typed) > 1:
        return typed, _edit(out, "\n".join(typed))
    return typed, typed[0]


def main() -> None:
    p = argparse.ArgumentParser(
        prog="bruh",
        description="A minimal coding agent over chat completions and messages APIs",
    )
    p.add_argument("-p", "--prompt", help="Run a single prompt and exit (non-interactive)")
    p.add_argument("--provider", choices=["openai", "anthropic"], default="openai",
                   help="API format: chat completions or messages")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Model id (default: {DEFAULT_MODEL}). Full list: https://opencode.ai/docs/go/ "
                        "(models served over /messages need --provider anthropic)")
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=None,
                   help="Sampling temperature; omitted from the request unless set "
                        "(some models, e.g. kimi-k3, only accept their own default)")
    p.add_argument("--max-steps", type=int, default=MAX_STEPS, help="Max tool-call steps per turn")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Show thinking, token estimate, and full tool output")
    p.add_argument("--rules", dest="include_rules", action="store_true",
                   help="Load AGENTS.md rule files (~/AGENTS.md and ./AGENTS.md)")
    p.add_argument("--no-skills", dest="include_skills", action="store_false",
                   help="Don't advertise skills from ~/.agents/skills and ./.agents/skills")
    args = p.parse_args()

    provider = _make_provider(args.provider)
    agent = Agent(
        provider=provider,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_steps=args.max_steps,
        verbose=args.verbose,
        include_rules=args.include_rules,
        include_skills=args.include_skills,
    )

    if args.prompt:
        agent.run(args.prompt)
        return

    agent.out.banner(f"bruh ready ({agent.model}). /edit for multi-line, ^D to exit.")
    while True:
        try:
            typed, text = _read_message(agent.out)
        except (EOFError, KeyboardInterrupt):
            agent.out.turn_break()
            agent.out.banner("bye.")
            return
        if not text.strip():
            continue
        agent.out.user_message(typed, text)
        agent.run(text)
        agent.out.turn_break()


if __name__ == "__main__":
    main()
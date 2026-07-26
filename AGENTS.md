# bruh

A minimal coding agent in Python. Talks to chat completions and messages APIs, driven by a tool-using planning loop.

Design and behavior are informed by [pi](https://pi.dev) (source at `../pi`, package
`@earendil-works/pi-agent-core`). bruh is a stripped-down Python port of the
ideas there, not a feature-for-feature clone; when in doubt about an intended
behavior, the corresponding pi code is the reference.

## Quick orientation

- ~1200 LOC, 4 modules under `src/bruh/`. Dependency: `httpx` only.
- Two streaming backends against the same base URL, selected via `--provider`:
  - `openai` (default) -> `{base}/chat/completions`
  - `anthropic`       -> `{base}/messages`
- Six tools the model can call autonomously: `bash`, `read`, `write`, `edit`, `grep`, `glob`.
- Tool calls within a single assistant turn run concurrently via a
  `ThreadPoolExecutor` (see `Agent._execute_tool_turn`); results are reordered
  back to the wire order before being appended (the Anthropic messages API
  requires tool_result ids to line up with the prior tool_use ids).
- Default base URL: `https://opencode.ai/zen/go/v1`. Override with `BRUH_BASE_URL`.
- API key from `BRUH_API_KEY`.

## Commands

```bash
uv sync                    # install httpx into .venv
uv run bruh --help
uv run bruh -v             # interactive, verbose (thinking + tools + token estimate to stderr)
uv run bruh -p "prompt"    # one-shot, non-interactive
uv run bruh --provider anthropic --model minimax-m3 -p "..."
```

No test suite yet. Verify changes by running the CLI and checking imports:

```bash
uv run python -W error -c "import bruh.agent, bruh.prompts, bruh.providers, bruh.tools; print('ok')"
uv run bruh --help
```

For a live smoke test (needs `BRUH_API_KEY`):

```bash
uv run bruh -p "Say exactly: ok"
```

## Architecture

```
src/bruh/
  __init__.py
  agent.py       Agent loop, argparse CLI
  prompts.py     System prompt assembly (base prompt, cwd, AGENTS.md context)
  providers.py   chat-completions + messages streaming clients, token estimation
  tools.py       Tool implementations + JSON-schema registry (TOOL_SPECS)
pyproject.toml   uv project; depends only on httpx
```

### Data flow

1. `agent.py:Agent.run()` appends a user `Message`, calls `_drive_loop()`.
2. `_drive_loop()` calls `_stream_step()` per turn:
   - Provider yields `Event` objects (`text`, `thinking`, `tool_start`, `tool_delta`, `tool_end`, `done`, `error`) by parsing SSE via httpx.
   - `_stream_step` accumulates them into `ContentBlock`s (`TextBlock`, `ThinkingBlock`, `ToolUseBlock`).
3. Each `ToolUseBlock` is executed via `tools.run_tool()`, producing a string.
4. Results are wrapped in `ToolResultBlock`s, appended as a `user` `Message`, and the loop repeats.
5. Loop ends when the assistant responds without tool calls, or `max_steps` is hit.

### Message types

Defined in `providers.py`:
- `TextBlock`, `ThinkingBlock`, `ToolUseBlock`, `ToolResultBlock` (role-union `ContentBlock`)
- `Message(role, content: list[ContentBlock])`

Both providers serialize `Message`s to their wire format in `_build_messages`.

### Token estimation

`estimate_tokens(message)` and `estimate_context_tokens(messages, system)` use a chars/4 heuristic (conservative overestimate). When a provider reports real usage in an `Event`, the agent anchors on that number (`Agent.last_usage`) and only estimates the *trailing* messages added since. Surfaced in verbose mode after each tool turn. No compaction logic; message history grows unbounded.

## Code conventions

- Python 3.12+, stdlib + httpx only. No other third-party deps.
- `from __future__ import annotations` at the top of every module.
- Type hints required. Dataclasses for value types. `Protocol` for the `Provider` interface.
- No comments unless explaining a non-obvious decision (the existing comment headers in `providers.py` are section dividers, not explanations).
- Tool functions in `tools.py` return a string. Errors start with `[error` or `[` prefix and are detected by `is_error=result.startswith("[error")` in the agent loop.
- `TOOL_SPECS` (list of dicts with `name`, `description`, `input_schema`) is the source of truth for what's advertised to the model; `TOOLS` maps name -> function.

## Known limitations (intentional, not bugs)

- No context compaction. Watch the verbose token estimate and start a new session when it gets large.
- Streamed thinking is shown in verbose mode but never stored in history.
- `MAX_TOOL_OUTPUT` (20_000 chars) caps each tool result before feeding back to the model.
- No permissions system; tools run with the launching user's privileges.
- A `KeyboardInterrupt` mid-stream aborts the current turn cleanly (drops
  in-flight tool calls, keeps any streamed text); the session continues.
  Bash subprocesses started by a tool are *not* cancelled — the turn's tool
  batch either already ran or didn't. See `Agent._stream_step`.

## Things NOT to do

- Don't add third-party deps beyond httpx without strong reason.
- Don't add a test framework unless asked; the project is small enough to verify by running.
- Don't add context compaction unless asked (intentionally omitted for simplicity).

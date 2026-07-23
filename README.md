# simple-agent

A small, pi-style coding agent in Python. Talks to the **opencode-go**
(`glm-5.2`) endpoint via OpenAI-completions or Anthropic-messages wire
format, driven by a tool-using planning loop.

Inspired by the architecture of [`pi`](../pi) but minimal: ~300 lines of
Python, stdlib-only except for [httpx](https://www.python-httpx.org/) for
streaming HTTP/SSE.

## What it does

- Streams model responses live to your terminal
- Six built-in tools the model can call autonomously:
  `bash`, `read`, `write`, `edit`, `grep`, `glob`
- Multi-step planning loop: explore -> plan -> act -> verify
- Two backend formats, selectable at the CLI (the endpoint depends on the
  model — see [opencode.ai/docs/go](https://opencode.ai/docs/go)):
  - `--provider openai`    -> `…/v1/chat/completions` (default; for glm-5.2,
    grok-4.5, kimi-*, deepseek-v4-*, mimo-v2.5*, hy3)
  - `--provider anthropic` -> `…/v1/messages` (for minimax-m3/m2.7/m2.5,
    qwen3.7-max/plus, qwen3.6-plus)

## Setup

```bash
uv sync                                    # installs httpx
export OPENCODE_API_KEY=...                # your opencode-go key
# optional: export OPENCODE_BASE_URL=https://opencode.ai/zen/go/v1
```

## Usage

Interactive session:

```bash
uv run simple-agent -v
you> /how big is src/main.py
you> add type hints to utils.py and run the tests
```

One-shot prompt, non-interactive:

```bash
uv run simple-agent -p "summarize the top-level layout of this repo"
```

Use the Anthropic wire format instead of OpenAI:

```bash
uv run simple-agent --provider anthropic --model minimax-m3 -p "find and fix the typo in README"
```

Flags:

```
-p, --prompt PROMPT               single prompt, then exit
--provider {openai,anthropic}     wire format (default: openai)
--model MODEL                     model id, e.g. glm-5.2, kimi-k3, minimax-m3
                                  (default: glm-5.2)
--max-tokens N                    completion cap (default: 8192)
--temperature T                   sampling temp (default: 0.7)
--max-steps N                     max tool-call rounds per turn (default: 50)
-v, --verbose                     stream thinking + tool calls to stderr
```

## Layout

```
src/simple_agent/
  __init__.py
  agent.py       Agent loop, argparse CLI, system prompt
  providers.py   OpenAI-completions + Anthropic-messages streaming clients
  tools.py       Tool implementations and the JSON-schema registry
pyproject.toml   uv project; depends only on httpx
```

## Notes / limitations

- Tool round-trips are synchronous; no parallel tool execution.
- The `ThinkingBlock` is recorded in the assistant turn but is only
  truly round-tripped by the Anthropic wire format.
- `MAX_TOOL_OUTPUT` (`20_000` chars) caps each tool result before it is
  fed back to the model, to keep history bounded.
- If the model calls a tool repeatedly without finishing,
  `--max-steps` stops the loop (default 50).
- Not affiliated with the `pi` project; this is an independent, simpler
  implementation that targets the same opencode-go endpoint.
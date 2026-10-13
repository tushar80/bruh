# bruh

A minimal coding agent in Python. Talks to chat completions and messages
APIs, driven by a tool-using planning loop. ~1700 lines of Python, depends
only on [httpx](https://www.python-httpx.org/) for streaming HTTP/SSE.
Licensed under the [MIT License](LICENSE).

Design and behavior are informed by [pi](https://pi.dev)
([source](https://github.com/earendil-works/pi)); bruh is a stripped-down Python port of the
ideas there, not a feature-for-feature clone.

## What it does

- Streams model responses live to your terminal
- Six built-in tools the model can call autonomously:
  `bash`, `read`, `write`, `edit`, `grep`, `glob`
- Multi-step planning loop: explore -> plan -> act -> verify
- Tool calls within a turn run in parallel
- Two backend formats, selectable at the CLI:
  - `--provider openai`    -> `{base}/chat/completions` (default)
  - `--provider anthropic` -> `{base}/messages`
- Live context-token estimate in verbose mode, anchored to real provider usage when available
- Agent skills: `SKILL.md` files under `~/.agents/skills` and `./.agents/skills` are listed in the
  system prompt, and the model reads one when the task matches
- Multi-line prompts: a paste, or `/edit`, opens in `$VISUAL`/`$EDITOR` before sending

## Setup

```bash
uv sync
export BRUH_API_KEY=...                # your api key
# optional: export BRUH_BASE_URL=https://your-endpoint/v1 (default: OpenCode Go)
```

## Usage

Interactive session:

```bash
uv run bruh -v
❯ how big is src/main.py
❯ add type hints to utils.py and run the tests
```

One-shot prompt, non-interactive:

```bash
uv run bruh -p "summarize the top-level layout of this repo"
```

Use the messages wire format instead of chat completions:

```bash
uv run bruh --provider anthropic --model minimax-m3 -p "find and fix the typo in README"
```

Flags:

```
-p, --prompt PROMPT               single prompt, then exit
--provider {openai,anthropic}     wire format (default: openai)
--model MODEL                     model id (default: deepseek-v4.1-flash)
--max-tokens N                    completion cap (default: 8192)
--temperature T                   sampling temp (default: omitted, model default)
--max-steps N                     max tool-call rounds per turn (default: 50)
-v, --verbose                     show thinking, token estimate, and full tool output
--rules                           load ~/AGENTS.md and ./AGENTS.md into the system prompt
--no-skills                       don't list skills from ~/.agents/skills and ./.agents/skills
```

## Layout

```
src/bruh/
  __init__.py
  agent.py       Agent loop, argparse CLI
  prompts.py     System prompt assembly (base prompt, cwd, skills index, AGENTS.md context)
  providers.py   chat-completions + messages streaming clients, token estimation
  render.py      Terminal presentation (ANSI theme, tool-call blocks)
  tools.py       Tool implementations and the JSON-schema registry
pyproject.toml   uv project; depends only on httpx
```

## Notes / limitations

- Tools within a turn run in parallel. Ctrl+C mid-turn kills running bash
  commands (and anything they started); other tools finish first.
- Streamed thinking is shown only in verbose mode. Anthropic thinking blocks
  stay in history because the messages API rejects a follow-up turn without them.
- Each tool caps its own output at 20,000 chars and says so when it truncates.
  bash keeps the tail and saves the full output to a temp file.
- If the model calls a tool repeatedly without finishing,
  `--max-steps` stops the loop (default 50).
- No context compaction: message history grows unbounded. Watch the
  verbose token estimate and start a new session when it gets large.

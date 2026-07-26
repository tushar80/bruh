"""System prompt assembly: base prompt, guidelines, project context."""

from __future__ import annotations

import os
from pathlib import Path

SYSTEM_PROMPT = """\
You are an expert coding assistant operating in the user's local repository.
You help users by reading files, executing commands, editing code, and writing
new files.

Work step by step:

1. Explore: read files, grep, glob, run bash to understand the task.
2. Plan: briefly outline your approach before editing.
3. Act: make edits with the write/edit tools.
4. Verify: run tests / lint / the relevant command before declaring done.

Guidelines:

- Use read to examine files instead of cat or sed
- Use grep and glob to search instead of running grep/find via bash
- Use write only for new files or complete rewrites; use edit for targeted changes
- Keep edit old_string as small as possible while staying unique in the file
- Make all independent tool calls in the same block when possible
- Be concise in your responses
- Show file paths clearly when working with files

When you have finished the task, respond with text (no tool call) describing
what you did.
"""

CONTEXT_FILE = "AGENTS.md"


def _context_files(cwd: Path) -> list[Path]:
    """AGENTS.md files: the home directory first, then cwd."""
    dirs = [Path.home(), cwd] if cwd != Path.home() else [cwd]
    return [d / CONTEXT_FILE for d in dirs if (d / CONTEXT_FILE).is_file()]


def build_system_prompt() -> str:
    cwd = Path(os.getcwd())
    prompt = SYSTEM_PROMPT + f"\nCurrent working directory: {cwd}\n"
    files = _context_files(cwd)
    if not files:
        return prompt
    blocks = [
        f'<project_instructions path="{p}">\n'
        f'{p.read_text(encoding="utf-8", errors="replace")}\n'
        f"</project_instructions>"
        for p in files
    ]
    prompt += "\n<project_context>\n\nProject-specific instructions and guidelines:\n\n"
    return prompt + "\n\n".join(blocks) + "\n\n</project_context>\n"

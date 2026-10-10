"""System prompt assembly: base prompt, guidelines, project context, skills."""

from __future__ import annotations

import html
import os
import re
from dataclasses import dataclass
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
SKILLS_DIR = Path(".agents") / "skills"
SKILL_FILE = "SKILL.md"


@dataclass
class Skill:
    name: str
    description: str
    path: Path


def _rule_files(cwd: Path) -> list[Path]:
    """Rule files (AGENTS.md): home directory first, then cwd."""
    dirs = [Path.home(), cwd] if cwd != Path.home() else [cwd]
    return [d / CONTEXT_FILE for d in dirs if (d / CONTEXT_FILE).is_file()]


def _frontmatter(text: str) -> dict[str, str]:
    """Top-level scalar keys from a YAML frontmatter block. Handles plain and
    quoted values and `|`/`>` block scalars, which covers SKILL.md in practice
    without pulling in a YAML parser."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    fields: dict[str, str] = {}
    key = ""
    block: list[str] | None = None
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if block is not None and (not line.strip() or line[:1] in " \t"):
            block.append(line.strip())
            continue
        if block is not None:
            fields[key] = " ".join(b for b in block if b)
            block = None
        m = re.match(r"([A-Za-z0-9_-]+):\s*(.*)$", line)
        if not m:
            continue
        key, value = m[1], m[2].strip()
        if value[:1] in "|>":
            block = []
        elif len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            fields[key] = value[1:-1]
        else:
            fields[key] = value
    if block is not None:
        fields[key] = " ".join(b for b in block if b)
    return fields


def load_skills(cwd: Path) -> list[Skill]:
    """Skills under ~/.agents/skills, then ./.agents/skills. A directory holding
    SKILL.md is a skill and is not searched further. On a name clash the first
    one found wins, matching pi."""
    roots = [Path.home() / SKILLS_DIR, cwd / SKILLS_DIR]
    skills: dict[str, Skill] = {}
    seen: set[Path] = set()
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d != "node_modules")
            if SKILL_FILE not in filenames:
                continue
            dirnames.clear()
            path = Path(dirpath) / SKILL_FILE
            real = path.resolve()
            if real in seen:
                continue
            seen.add(real)
            meta = _frontmatter(path.read_text(encoding="utf-8", errors="replace"))
            name = meta.get("name") or path.parent.name
            if meta.get("description") and name not in skills:
                skills[name] = Skill(name, meta["description"], path.absolute())
    return list(skills.values())


def _skills_section(skills: list[Skill]) -> str:
    entries = [
        f"  <skill>\n"
        f"    <name>{html.escape(s.name)}</name>\n"
        f"    <description>{html.escape(s.description)}</description>\n"
        f"    <location>{html.escape(str(s.path))}</location>\n"
        f"  </skill>"
        for s in skills
    ]
    return (
        "\nThe following skills provide specialized instructions for specific tasks.\n"
        "Use the read tool to load a skill's file when the task matches its description.\n"
        "When a skill file references a relative path, resolve it against the skill directory "
        "(the directory containing SKILL.md) and use that absolute path in tool calls.\n\n"
        "<available_skills>\n" + "\n".join(entries) + "\n</available_skills>\n"
    )


def build_system_prompt(*, include_rules: bool = False, include_skills: bool = True) -> str:
    cwd = Path.cwd()
    prompt = SYSTEM_PROMPT + f"\nCurrent working directory: {cwd}\n"
    if include_skills and (skills := load_skills(cwd)):
        prompt += _skills_section(skills)
    if not include_rules:
        return prompt
    files = _rule_files(cwd)
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

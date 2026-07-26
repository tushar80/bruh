"""Tools the agent can call: bash, read, write, edit, grep, glob.

All tools return a string result. Tool schemas (JSON-schema dicts) live in
``TOOL_SPECS`` so providers can advertise them to the model.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any
from collections.abc import Callable

MAX_READ_LINES = 2000
MAX_OUTPUT_CHARS = 20_000
MAX_RESULTS = 50

ToolFn = Callable[..., str]


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    kept = text[:limit]
    return kept + f"\n...[truncated {len(text) - limit} chars]"


def tool_bash(command: str, workdir: str | None = None, timeout: int = 120) -> str:
    """Run a shell command and return combined stdout+stderr."""
    cwd = workdir or os.getcwd()
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0:
            out += f"\n[exit code {proc.returncode}]"
        return _truncate(out) if out else f"[exit code {proc.returncode}, no output]"
    except subprocess.TimeoutExpired:
        return f"[command timed out after {timeout}s]"
    except Exception as exc:  # noqa: BLE001
        return f"[error running command: {exc}]"


def tool_read(file_path: str, offset: int = 1, limit: int = MAX_READ_LINES) -> str:
    """Read a slice of a file, 1-indexed line numbers, with line-number prefixes."""
    try:
        p = Path(file_path)
        if not p.exists() or not p.is_file():
            return f"[error: file not found: {file_path}]"
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, offset)
        end = min(len(lines), start + limit - 1)
        buf: list[str] = []
        for i in range(start, end + 1):
            line = lines[i - 1]
            if len(line) > 2000:
                line = line[:2000] + "...[truncated]"
            buf.append(f"{i}: {line}")
        if not buf:
            return "[file is empty]"
        return "\n".join(buf)
    except Exception as exc:  # noqa: BLE001
        return f"[error reading file: {exc}]"


def tool_write(file_path: str, content: str) -> str:
    """Write content to a file (creates or overwrites)."""
    try:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"[wrote {len(content)} bytes to {file_path}]"
    except Exception as exc:  # noqa: BLE001
        return f"[error writing file: {exc}]"


def tool_edit(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Replace an exact string match in a file. Fails if old_string not found
    or (when replace_all=False) if it matches more than once."""
    try:
        p = Path(file_path)
        if not p.exists() or not p.is_file():
            return f"[error: file not found: {file_path}]"
        text = p.read_text(encoding="utf-8", errors="replace")
        count = text.count(old_string)
        if count == 0:
            return "[error: old_string not found in file]"
        if not replace_all and count > 1:
            return f"[error: old_string found {count} times; pass replace_all=true or add more context]"
        if replace_all:
            new_text = text.replace(old_string, new_string)
        else:
            new_text = text.replace(old_string, new_string, 1)
        p.write_text(new_text, encoding="utf-8")
        return f"[edited {file_path}: {count} replacement(s)]"
    except Exception as exc:  # noqa: BLE001
        return f"[error editing file: {exc}]"


def tool_grep(pattern: str, path: str = ".", include: str | None = None) -> str:
    """Search file contents with a regex. Returns matches as path:line:content."""
    try:
        regex = re.compile(pattern)
        root = Path(path)
        if root.is_file():
            files = [root]
        else:
            files = [p for p in root.rglob("*") if p.is_file()]
        buf: list[str] = []
        matches = 0
        for fp in files:
            if include and not fnmatch.fnmatch(fp.name, include):
                continue
            if ".git" in fp.parts:
                continue
            try:
                for lineno, line in enumerate(
                    fp.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
                ):
                    if regex.search(line):
                        buf.append(f"{fp}:{lineno}: {line[:2000]}")
                        matches += 1
                        if matches >= MAX_RESULTS:
                            buf.append(f"...[max {MAX_RESULTS} results reached]")
                            return "\n".join(buf)
            except Exception:
                continue
        if not buf:
            return "[no matches]"
        return "\n".join(buf)
    except re.error as exc:
        return f"[invalid regex: {exc}]"
    except Exception as exc:  # noqa: BLE001
        return f"[error grepping: {exc}]"


def tool_glob(pattern: str, path: str = ".") -> str:
    """Match file paths by glob pattern."""
    try:
        root = Path(path)
        results = sorted(p for p in root.glob(pattern) if ".git" not in p.parts)
        # Also support recursive ** when not present
        if not results and "**" not in pattern:
            results = sorted(p for p in root.rglob(pattern) if ".git" not in p.parts)
        if not results:
            return "[no matches]"
        out = [str(p) for p in results[:MAX_RESULTS]]
        if len(results) > MAX_RESULTS:
            out.append(f"...[{len(results)} matches, showing first {MAX_RESULTS}]")
        return "\n".join(out)
    except Exception as exc:  # noqa: BLE001
        return f"[error globbing: {exc}]"


# --- Registry ---------------------------------------------------------------

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "bash",
        "description": f"Run a shell command. Returns stdout+stderr, truncated to {MAX_OUTPUT_CHARS} chars. Use for running tests, git, builds, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute."},
                "workdir": {"type": "string", "description": "Working directory (default: current directory)."},
                "timeout": {"type": "integer", "description": "Timeout in seconds.", "default": 120},
            },
            "required": ["command"],
        },
    },
    {
        "name": "read",
        "description": f"Read a file. Returns lines prefixed with 1-indexed line numbers, at most {MAX_READ_LINES} lines per call. Use offset/limit to page through large files until complete.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file (relative or absolute)."},
                "offset": {"type": "integer", "description": "1-indexed line to start at.", "default": 1},
                "limit": {"type": "integer", "description": "Max lines to read.", "default": MAX_READ_LINES},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "write",
        "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories. Use only for new files or complete rewrites; use edit for targeted changes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file (relative or absolute)."},
                "content": {"type": "string", "description": "Content to write."},
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "edit",
        "description": "Replace an exact string in a file. old_string must match exactly and be unique in the file unless replace_all is set. Keep old_string as small as possible while staying unique; add surrounding context to disambiguate.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file (relative or absolute)."},
                "old_string": {"type": "string", "description": "Exact text to replace."},
                "new_string": {"type": "string", "description": "Replacement text."},
                "replace_all": {"type": "boolean", "description": "Replace all occurrences.", "default": False},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
    {
        "name": "grep",
        "description": f"Search file contents with a regex. Returns matches as path:line:content, limited to {MAX_RESULTS} matches.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex to search for."},
                "path": {"type": "string", "description": "Directory or file to search.", "default": "."},
                "include": {"type": "string", "description": "Filename glob filter, e.g. '*.py'."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "glob",
        "description": f"Find files by glob pattern (e.g. 'src/**/*.py'), limited to {MAX_RESULTS} results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern to match."},
                "path": {"type": "string", "description": "Directory to search in.", "default": "."},
            },
            "required": ["pattern"],
        },
    },
]

TOOLS: dict[str, ToolFn] = {
    "bash": tool_bash,
    "read": tool_read,
    "write": tool_write,
    "edit": tool_edit,
    "grep": tool_grep,
    "glob": tool_glob,
}


def run_tool(name: str, arguments: dict[str, Any]) -> str:
    fn = TOOLS.get(name)
    if fn is None:
        return f"[unknown tool: {name}]"
    try:
        return fn(**arguments)
    except TypeError as exc:
        return f"[bad arguments for {name}: {exc}]"
    except Exception as exc:  # noqa: BLE001
        return f"[tool {name} crashed: {exc}]"
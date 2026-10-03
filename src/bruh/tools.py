"""Tools the agent can call: bash, read, write, edit, grep, glob.

All tools return a string result. Tool schemas (JSON-schema dicts) live in
``TOOL_SPECS`` so providers can advertise them to the model.
"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from collections.abc import Callable

MAX_READ_LINES = 2000
MAX_OUTPUT_CHARS = 20_000
MAX_RESULTS = 50
SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules"}
_SKIP_TEXT = ", ".join(sorted(SKIP_DIRS))

ToolFn = Callable[..., str]


def _truncate_tail(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Keep the tail of long command output, where errors usually are; the
    full text is saved to a temp file so it can be read or grepped."""
    if len(text) <= limit:
        return text
    fd, path = tempfile.mkstemp(prefix="bruh-", suffix=".log", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    kept = text[-limit:]
    return f"[...truncated {len(text) - limit} chars; full output saved to {path}]\n{kept}"


def tool_bash(command: str, workdir: str | None = None, timeout: int = 120) -> str:
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=workdir or os.getcwd(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"[error: command timed out after {timeout}s]"
    out = _truncate_tail(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        return f"[error: exit code {proc.returncode}]\n{out}"
    return out or "[no output]"


def tool_read(file_path: str, offset: int = 1, limit: int = MAX_READ_LINES) -> str:
    p = Path(file_path)
    if not p.is_file():
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
    text = "\n".join(buf)
    if len(text) > MAX_OUTPUT_CHARS:
        text = text[:MAX_OUTPUT_CHARS] + "\n...[truncated; use offset to continue]"
    return text


def tool_write(file_path: str, content: str) -> str:
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"[wrote {len(content)} bytes to {file_path}]"


def tool_edit(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    p = Path(file_path)
    if not p.is_file():
        return f"[error: file not found: {file_path}]"
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"[error: {file_path} is not valid UTF-8; edit would corrupt it]"
    count = text.count(old_string)
    if count == 0:
        return "[error: old_string not found in file]"
    if not replace_all and count > 1:
        return f"[error: old_string found {count} times; pass replace_all=true or add more context]"
    p.write_text(text.replace(old_string, new_string, -1 if replace_all else 1), encoding="utf-8")
    return f"[edited {file_path}: {count} replacement(s)]"


def tool_grep(
    pattern: str,
    path: str = ".",
    include: str | None = None,
    ignore_case: bool = False,
    literal: bool = False,
    limit: int = MAX_RESULTS,
) -> str:
    try:
        regex = re.compile(
            re.escape(pattern) if literal else pattern,
            re.IGNORECASE if ignore_case else 0,
        )
    except re.error as exc:
        return f"[error: invalid regex: {exc}]"
    root = Path(path)
    files: list[Path] = []
    if root.is_file():
        files = [root]
    else:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            files.extend(Path(dirpath) / name for name in filenames)
    buf: list[str] = []
    size = 0
    for fp in files:
        if include and not fnmatch.fnmatch(fp.name, include):
            continue
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if regex.search(line):
                buf.append(f"{fp}:{lineno}: {line[:2000]}")
                size += len(buf[-1]) + 1
                if size > MAX_OUTPUT_CHARS:
                    buf.append(f"...[truncated at {MAX_OUTPUT_CHARS} chars; narrow the pattern or path]")
                    return "\n".join(buf)
                if len(buf) >= limit:
                    buf.append(f"...[max {limit} results reached]")
                    return "\n".join(buf)
    if not buf:
        return "[no matches]"
    return "\n".join(buf)


def tool_glob(pattern: str, path: str = ".") -> str:
    root = Path(path)
    results = sorted(p for p in root.glob(pattern) if not SKIP_DIRS & set(p.parts))
    if not results and "**" not in pattern:
        results = sorted(p for p in root.rglob(pattern) if not SKIP_DIRS & set(p.parts))
    if not results:
        return "[no matches]"
    out = [str(p) for p in results[:MAX_RESULTS]]
    if len(results) > MAX_RESULTS:
        out.append(f"...[{len(results)} matches, showing first {MAX_RESULTS}]")
    return "\n".join(out)


TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "bash",
        "description": f"Run a shell command. Returns stdout+stderr; output longer than {MAX_OUTPUT_CHARS} chars keeps the tail and saves the full output to a temp file. Use for running tests, git, builds, etc.",
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
        "description": f"Read a file. Returns lines prefixed with 1-indexed line numbers, at most {MAX_READ_LINES} lines or {MAX_OUTPUT_CHARS} chars per call. Use offset/limit to page through large files until complete.",
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
        "description": f"Search file contents with a regex. Returns matches as path:line:content, limited to {MAX_RESULTS} matches by default. Skips {_SKIP_TEXT}.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex to search for (literal text if literal is true)."},
                "path": {"type": "string", "description": "Directory or file to search.", "default": "."},
                "include": {"type": "string", "description": "Filename glob filter, e.g. '*.py'."},
                "ignore_case": {"type": "boolean", "description": "Case-insensitive search.", "default": False},
                "literal": {"type": "boolean", "description": "Treat pattern as literal text instead of regex.", "default": False},
                "limit": {"type": "integer", "description": "Max matches to return.", "default": MAX_RESULTS},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "glob",
        "description": f"Find files by glob pattern (e.g. 'src/**/*.py'), limited to {MAX_RESULTS} results. Skips {_SKIP_TEXT}.",
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
        return f"[error: unknown tool {name}]"
    try:
        return fn(**arguments)
    except TypeError as exc:
        return f"[error: bad arguments for {name}: {exc}]"
    except Exception as exc:
        return f"[error: {name} failed: {exc}]"
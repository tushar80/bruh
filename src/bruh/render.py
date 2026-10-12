"""Terminal presentation: ANSI theme, tool-call blocks, stream boundaries.

Everything a human reads is formatted here and written to stderr; stdout carries
only model text, so ``bruh -p ... > out.txt`` stays clean.

Colors follow pi's semantics: ``fg`` resets with ``39`` and ``bg`` with ``49``,
never with ``0``, so a colored line inside a tinted block does not wipe the
block's background mid-row.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import textwrap
import threading
from typing import Any

_TRUECOLOR: dict[str, str | int] = {
    "header_bg": "#282832",
    "user_bg": "#2e3754",
    "ok_bg": "#283228",
    "err_bg": "#3c2828",
    "title": "#d4d4d4",
    "accent": "#8abeb7",
    "muted": "#808080",
    "dim": "#666666",
    "warning": "#f0c674",
    "error": "#cc6666",
    "thinking": "#808080",
}

_XTERM256: dict[str, str | int] = {
    "header_bg": 236,
    "user_bg": 17,
    "ok_bg": 22,
    "err_bg": 52,
    "title": 252,
    "accent": 109,
    "muted": 244,
    "dim": 240,
    "warning": 179,
    "error": 167,
    "thinking": 244,
}

_BG_TOKENS = ("header_bg", "user_bg", "ok_bg", "err_bg")


def _color_enabled() -> bool:
    force = os.environ.get("FORCE_COLOR")
    if force is not None:
        # Node's convention: 0 means off, any level above that means on.
        return force not in ("", "0", "false")
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return sys.stderr.isatty()


def _sgr(color: str | int, base: int) -> str:
    """base is 38 for foreground, 48 for background."""
    if isinstance(color, int):
        return f"\x1b[{base};5;{color}m"
    r, g, b = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return f"\x1b[{base};2;{r};{g};{b}m"


COLOR = _color_enabled()
_PALETTE = (
    _TRUECOLOR if os.environ.get("COLORTERM") in ("truecolor", "24bit") else _XTERM256
)
_FG = {token: _sgr(color, 38) for token, color in _PALETTE.items()} if COLOR else {}
_BG = {token: _sgr(_PALETTE[token], 48) for token in _BG_TOKENS} if COLOR else {}


def fg(token: str, text: str) -> str:
    code = _FG.get(token)
    return f"{code}{text}\x1b[39m" if code else text


def bg(token: str, text: str) -> str:
    code = _BG.get(token)
    return f"{code}{text}\x1b[49m" if code else text


def bold(text: str) -> str:
    return f"\x1b[1m{text}\x1b[22m" if COLOR else text


def italic(text: str) -> str:
    return f"\x1b[3m{text}\x1b[23m" if COLOR else text


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_RESET = "\x1b[39m\x1b[22m\x1b[23m"
_INVALID = fg("error", "[invalid arg]")

# These mirror return strings in tools.py; changing those means changing these.
_EXIT_RE = re.compile(r"^\[error: exit code (\d+)\]\n?")
_ERROR_RE = re.compile(r"^\[error:? ?(.*)\]$")
_WROTE_RE = re.compile(r"^\[wrote (\d+ bytes) to .*\]$")
_EDITED_RE = re.compile(r"^\[edited .*: (\d+ replacement\(s\))\]$")
_TRUNC_RE = re.compile(r"^(\.\.\.)?\[(\.\.\.)?(truncated|max \d+|\d+ matches)")

# pi hides read results until expanded; bruh has no expand, so the header
# (path + range) stands alone and -v is the way to see the contents.
HIDE_BODY = ("read",)
HEAD_ONLY = ("write", "edit")  # result collapses to a one-line summary
TAIL_PREVIEW = ("bash",)  # keep the end of the output, where failures are
PREVIEW_LINES = {"bash": 5}
DEFAULT_PREVIEW = 10
PROMPT_WIDTH = 2  # visible width of prompt() and continuation()


def _width() -> int:
    return max(24, shutil.get_terminal_size((80, 24)).columns)


def _vlen(text: str) -> int:
    return len(_ANSI_RE.sub("", text))


def _sanitize(line: str) -> str:
    return line.replace("\t", "    ").rstrip()


def _clip(text: str, limit: int) -> str:
    """Truncate to a visible width. Escape sequences up to the cut are kept, so
    the ellipsis inherits the style of the text it replaces and nothing leaks
    past it."""
    if _vlen(text) <= limit:
        return text
    keep = max(0, limit - 3)
    out: list[str] = []
    visible = 0
    pos = 0
    for match in _ANSI_RE.finditer(text):
        plain = text[pos : match.start()]
        if visible + len(plain) >= keep:
            out.append(plain[: keep - visible])
            break
        out.append(plain)
        visible += len(plain)
        out.append(match.group(0))
        pos = match.end()
    else:
        out.append(text[pos:][: keep - visible])
    return "".join(out) + "..." + (_RESET if COLOR else "")


def _row(content: str, tint: str, width: int, indent: bool = False) -> str:
    """One full-width tinted row. Without color the tint carries no information,
    so rows become bare lines that a redirected stderr can be read as a log;
    ``indent`` is what keeps result lines under their header there."""
    content = _clip(content, width - 2)
    if not COLOR:
        return ("  " if indent else "") + content
    padded = " " + content + " "
    return bg(tint, padded + " " * max(0, width - _vlen(padded)))


def _shorten(path: str) -> str:
    relative = os.path.relpath(path, os.getcwd())
    if relative != ".." and not relative.startswith(".." + os.sep):
        return relative
    home = os.path.expanduser("~")
    return "~" + path[len(home) :] if path.startswith(home + os.sep) else path


def _path_arg(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value:
        return _INVALID
    return fg("accent", _shorten(value))


def _header(name: str, args: dict[str, Any]) -> str:
    verb = bold(fg("title", name))

    if name == "bash":
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            return f"{bold(fg('title', '$'))} {_INVALID}"
        head = bold(fg("title", "$ " + " ".join(command.split())))
        timeout = args.get("timeout")
        if isinstance(timeout, int) and timeout != 120:
            head += fg("muted", f" (timeout {timeout}s)")
        workdir = args.get("workdir")
        if isinstance(workdir, str) and workdir:
            head += fg("muted", f" in {_shorten(workdir)}")
        return head

    if name == "read":
        head = f"{verb} {_path_arg(args, 'file_path')}"
        offset = args.get("offset")
        if isinstance(offset, int) and offset > 1:
            limit = args.get("limit")
            end = offset + limit - 1 if isinstance(limit, int) else ""
            head += fg("warning", f":{offset}-{end}")
        return head

    if name == "write":
        return f"{verb} {_path_arg(args, 'file_path')}"

    if name == "edit":
        head = f"{verb} {_path_arg(args, 'file_path')}"
        return head + fg("muted", " (all)") if args.get("replace_all") else head

    if name in ("grep", "glob"):
        pattern = args.get("pattern")
        if not isinstance(pattern, str):
            return f"{verb} {_INVALID}"
        head = f"{verb} {fg('accent', f'/{pattern}/' if name == 'grep' else pattern)}"
        path = args.get("path")
        if isinstance(path, str) and path not in ("", "."):
            head += fg("muted", f" in {_shorten(path)}")
        include = args.get("include")
        if isinstance(include, str) and include:
            head += fg("muted", f" ({include})")
        if args.get("ignore_case"):
            head += fg("muted", " -i")
        return head

    return f"{verb} {fg('muted', json.dumps(args, separators=(',', ':'), default=str))}"


def _summary(name: str, body: str) -> str:
    """Collapse a write/edit result to one line. The path is already in the
    header, so only the part that varies survives."""
    text = body.strip()
    match = (_WROTE_RE if name == "write" else _EDITED_RE).match(text)
    return match.group(1) if match else text


def _peel_truncation(body: str) -> tuple[str, str]:
    """Lift a truncation marker out of the body. bash writes its marker on the
    first line but its preview keeps the tail; grep and glob append theirs on
    the last line but their previews keep the head. At either end the marker
    would be dropped along with the lines the preview hides."""
    lines = body.splitlines()
    if lines and _TRUNC_RE.match(lines[0].strip()):
        return "\n".join(lines[1:]), lines[0].strip()
    if lines and _TRUNC_RE.match(lines[-1].strip()):
        return "\n".join(lines[:-1]), lines[-1].strip()
    return body, ""


def _preview(name: str, body: str, is_error: bool, verbose: bool) -> tuple[list[str], str]:
    """The lines worth showing, plus a note for what was left out."""
    lines = body.splitlines() or [""]
    if verbose:
        return lines, ""
    if not is_error and name in HIDE_BODY:
        return [], ""
    if not is_error and name in HEAD_ONLY:
        return [_summary(name, body)], ""
    budget = DEFAULT_PREVIEW if is_error else PREVIEW_LINES.get(name, DEFAULT_PREVIEW)
    hidden = len(lines) - budget
    if hidden <= 0:
        return lines, ""
    if name in TAIL_PREVIEW:
        return lines[-budget:], f"... ({hidden} earlier lines)"
    return lines[:budget], f"... ({hidden} more lines)"


def _block(
    name: str,
    args: dict[str, Any],
    result: str,
    elapsed: float,
    verbose: bool,
    width: int,
    with_header: bool,
) -> list[str]:
    """Rows for one finished tool call: header, result preview, what the model
    had truncated from it, then exit code and elapsed time."""
    is_error = result.startswith("[error")
    tint = "err_bg" if is_error else "ok_bg"

    exit_code = _EXIT_RE.match(result)
    body = result[exit_code.end() :] if exit_code else result
    body, truncated = _peel_truncation(body)
    if is_error:
        unwrapped = _ERROR_RE.match(body.strip())
        if unwrapped:
            body = unwrapped.group(1)

    footer: list[str] = []
    if exit_code:
        footer.append(f"exit {exit_code.group(1)}")
    if exit_code or elapsed >= 0.1:
        footer.append(f"{elapsed:.1f}s")

    lines, elided = _preview(name, body, is_error, verbose)
    # A lone summary line leaves room for the timing beside it. Errors keep their
    # own row so the message stays the only red thing in the block.
    if footer and not is_error and len(lines) == 1 and not elided and not truncated:
        lines = [f"{lines[0]}  {'  '.join(footer)}".strip()]
        footer = []

    def row(content: str) -> str:
        return _row(content, tint, width, indent=True)

    rows = [_row(_header(name, args), "header_bg", width)] if with_header else []
    if elided and name in TAIL_PREVIEW:
        rows.append(row(fg("muted", elided)))
    for line in lines:
        token = "warning" if _TRUNC_RE.match(line.strip()) else "error" if is_error else "muted"
        rows.append(row(fg(token, _sanitize(line))))
    if elided and name not in TAIL_PREVIEW:
        rows.append(row(fg("muted", elided)))
    if truncated:
        rows.append(row(fg("warning", truncated)))
    if footer:
        rows.append(row(fg("muted", "  ".join(footer))))
    return rows


class Renderer:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._lock = threading.Lock()
        self._thinking = False
        self._text_open = False

    def _err(self, text: str) -> None:
        sys.stdout.flush()
        sys.stderr.write(text)
        sys.stderr.flush()

    def _out(self, text: str) -> None:
        sys.stderr.flush()
        sys.stdout.write(text)
        sys.stdout.flush()

    def _close_thinking(self) -> bool:
        """Terminate a streamed reasoning run. Reasoning is written with no
        trailing newline, so without this the answer starts mid-line."""
        if not self._thinking:
            return False
        self._err("\n\n")
        self._thinking = False
        return True

    def _line(self, token: str, text: str, blank_before: bool = True) -> None:
        with self._lock:
            self._close_thinking()
            self._err(("\n" if blank_before else "") + fg(token, text) + "\n")

    def thinking(self, text: str) -> None:
        if not text or not self.verbose:
            return
        with self._lock:
            if not self._thinking:
                self._err("\n" + fg("dim", italic("thinking")) + "\n")
                self._thinking = True
            self._err(fg("thinking", italic(text)))

    def text(self, chunk: str) -> None:
        if not chunk:
            return
        with self._lock:
            closed = self._close_thinking()
            if not self._text_open:
                if not closed:
                    self._err("\n")
                self._text_open = True
            self._out(chunk)

    def end_text(self) -> None:
        with self._lock:
            self._close_thinking()
            if self._text_open:
                self._out("\n")
                self._text_open = False

    def tool_header(self, name: str, args: dict[str, Any]) -> None:
        row = _row(_header(name, args), "header_bg", _width())
        with self._lock:
            self._close_thinking()
            self._err("\n" + row + "\n")

    def tool_result(
        self,
        name: str,
        args: dict[str, Any],
        result: str,
        elapsed: float,
        header_done: bool,
    ) -> None:
        rows = _block(
            name, args, result, elapsed, self.verbose, _width(), not header_done
        )
        with self._lock:
            self._close_thinking()
            self._err(("" if header_done else "\n") + "\n".join(rows) + "\n")

    def parallel_note(self, count: int) -> None:
        self._line("dim", f"{count} tools in parallel")

    def banner(self, text: str) -> None:
        self._line("dim", text, blank_before=False)

    def prompt(self) -> str:
        if not COLOR or not sys.stdout.isatty():
            return "> "
        # input() goes through readline (libedit in uv's Python), which drops a
        # bare ESC from the prompt. \x01..\x02 marks a span as non-printing.
        return re.sub(r"\x1b\[[0-9;]*m", "\x01\\g<0>\x02", bold(fg("accent", "\u276f"))) + " "

    def continuation(self) -> str:
        return " " * PROMPT_WIDTH

    def user_message(self, typed: list[str], text: str) -> None:
        """Replace the echoed input with a tinted block, as pi does, so user
        turns stand apart from the model's answers. ``typed`` is what readline
        echoed; ``text`` can differ after a trip through the editor."""
        if not COLOR or not sys.stdout.isatty():
            return
        width = _width()
        echoed = sum(max(1, -(-(PROMPT_WIDTH + len(line)) // width)) for line in typed)
        rows = [_row("", "user_bg", width)]
        for line in text.splitlines():
            parts = textwrap.wrap(_sanitize(line), width - 2) or [""]
            rows += [_row(part, "user_bg", width) for part in parts]
        rows.append(_row("", "user_bg", width))
        with self._lock:
            # Cursor up over what readline echoed, then clear to end of screen.
            self._err(f"\x1b[{echoed}F\x1b[J" + "\n".join(rows) + "\n")

    def error(self, message: str) -> None:
        self._line("error", f"[error] {message}")

    def warn(self, message: str) -> None:
        self._line("warning", f"[{message}]")

    def debug(self, message: str) -> None:
        if self.verbose:
            self._line("dim", message)

    def tokens(self, estimated: int, real: int | None) -> None:
        suffix = f" (real: {real})" if real is not None else ""
        self.debug(f"~{estimated} tokens in context{suffix}")

    def turn_break(self) -> None:
        self._err("\n")

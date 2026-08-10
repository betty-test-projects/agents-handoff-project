"""
Filesystem tools for agents — read_file, list_files, write_file.

These tools operate on a sandboxed project root (the task_manager
directory). All paths agents pass in are resolved relative to that root,
and any path that would resolve outside the root is rejected.

Writes also trigger a snapshot of the entire project root, copied into
the run's snapshots directory. This lets us reconstruct what the code
looked like at every modification point during analysis.
"""

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


MAX_FILE_BYTES = 100 * 1024  # 100 KB read limit


# ── Tool schemas (the JSON descriptions exposed to the model) ────────────────

READ_FILE_SCHEMA = {
    "name": "read_file",
    "description": "Read the contents of a file at the given path.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file, relative to the project root.",
            },
        },
        "required": ["path"],
    },
}

LIST_FILES_SCHEMA = {
    "name": "list_files",
    "description": "List files and directories at the given path.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path, relative to the project root. Use '.' for the root.",
            },
        },
        "required": ["path"],
    },
}

WRITE_FILE_SCHEMA = {
    "name": "write_file",
    "description": "Write content to a file at the given path. Creates the file if it does not exist, overwrites if it does.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file, relative to the project root.",
            },
            "content": {
                "type": "string",
                "description": "Full content of the file.",
            },
        },
        "required": ["path", "content"],
    },
}


# ── Implementation ───────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_within_root(root: Path, user_path: str) -> Path:
    """
    Resolve a user-supplied path against the sandbox root.

    Returns the resolved absolute path if it stays within the root.
    Raises ValueError if the resolved path escapes the root.
    """
    if not user_path:
        raise ValueError("path is empty")

    # Reject absolute paths outright — agents should only use relative paths.
    candidate = Path(user_path)
    if candidate.is_absolute():
        raise ValueError(f"path '{user_path}' is absolute; only relative paths are allowed")

    # Resolve against the root, then check containment.
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        raise ValueError(f"path '{user_path}' resolves outside the project root")

    return resolved


@dataclass
class FilesystemTools:
    """
    Tool executor for filesystem operations.

    project_root: absolute path to the task_manager directory (the sandbox).
    snapshots_dir: where to write code snapshots on every successful write.
    log_event: optional callback invoked when a snapshot is taken; receives
        a dict describing the snapshot event so the agent's internal log
        can record it.
    """

    project_root: Path
    snapshots_dir: Path
    log_event: Callable | None = None
    _snapshot_counter: int = field(default=0, init=False)

    def __post_init__(self):
        self.project_root = Path(self.project_root)
        self.snapshots_dir = Path(self.snapshots_dir)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    # ── Public entry point ───────────────────────────────────────────────────

    def execute(self, tool_name: str, tool_input: dict) -> tuple[str, bool]:
        """Dispatch to the right tool by name."""
        if tool_name == "read_file":
            return self._read_file(tool_input)
        if tool_name == "list_files":
            return self._list_files(tool_input)
        if tool_name == "write_file":
            return self._write_file(tool_input)
        return (f"Error: unknown tool '{tool_name}'", False)

    # ── Individual tools ─────────────────────────────────────────────────────

    def _read_file(self, tool_input: dict) -> tuple[str, bool]:
        user_path = tool_input.get("path", "")
        try:
            full_path = _resolve_within_root(self.project_root, user_path)
        except ValueError as e:
            return (f"Error: {e}", False)

        if not full_path.exists():
            return (f"Error: file '{user_path}' does not exist", False)

        if not full_path.is_file():
            return (f"Error: '{user_path}' is not a file", False)

        size = full_path.stat().st_size
        if size > MAX_FILE_BYTES:
            return (
                f"Error: file '{user_path}' is {size} bytes, exceeds the "
                f"{MAX_FILE_BYTES}-byte read limit",
                False,
            )

        try:
            content = full_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return (f"Error: file '{user_path}' is not a UTF-8 text file", False)

        return (content, True)

    def _list_files(self, tool_input: dict) -> tuple[str, bool]:
        user_path = tool_input.get("path", "")
        try:
            full_path = _resolve_within_root(self.project_root, user_path)
        except ValueError as e:
            return (f"Error: {e}", False)

        if not full_path.exists():
            return (f"Error: path '{user_path}' does not exist", False)

        if not full_path.is_dir():
            return (f"Error: '{user_path}' is not a directory", False)

        entries = []
        for child in sorted(full_path.iterdir()):
            name = child.name
            if child.is_dir():
                entries.append(f"{name}/")
            else:
                entries.append(name)

        if not entries:
            return ("(empty directory)", True)

        return ("\n".join(entries), True)

    def _write_file(self, tool_input: dict) -> tuple[str, bool]:
        user_path = tool_input.get("path", "")
        content = tool_input.get("content", "")

        try:
            full_path = _resolve_within_root(self.project_root, user_path)
        except ValueError as e:
            return (f"Error: {e}", False)

        # Refuse to write to a directory path.
        if full_path.exists() and full_path.is_dir():
            return (f"Error: '{user_path}' is a directory, not a file", False)

        # Create parent directories if needed.
        full_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            full_path.write_text(content, encoding="utf-8")
        except OSError as e:
            return (f"Error: failed to write '{user_path}': {e}", False)

        # Snapshot after a successful write.
        self._take_snapshot(triggered_by_path=user_path)

        return ("OK", True)

    # ── Snapshot mechanism ───────────────────────────────────────────────────

    def _take_snapshot(self, *, triggered_by_path: str) -> None:
        """
        Copy the entire project root to snapshots_dir/snapshot_NNN/.

        Writes a meta.json into the snapshot describing what triggered it,
        and invokes log_event (if provided) so the agent's internal log
        can record this event too.
        """
        existing_nums = []
        for p in self.snapshots_dir.glob("snapshot_*"):
            if p.is_dir():
                stem = p.name.removeprefix("snapshot_")
                if stem.isdigit():
                    existing_nums.append(int(stem))
        next_num = max(existing_nums, default=0) + 1
        self._snapshot_counter = next_num
        snapshot_id = f"snapshot_{self._snapshot_counter:03d}"
        snapshot_path = self.snapshots_dir / snapshot_id

        # Copy the project root.
        shutil.copytree(self.project_root, snapshot_path)

        # Write meta.json into the snapshot.
        meta = {
            "snapshot_id": snapshot_id,
            "timestamp": _now_iso(),
            "triggered_by": {
                "tool": "write_file",
                "path": triggered_by_path,
            },
        }
        (snapshot_path / "meta.json").write_text(
            json.dumps(meta, indent=2) + "\n"
        )

        # Notify the runner's internal log if a callback is provided.
        if self.log_event is not None:
            self.log_event({
                "event": "snapshot",
                "timestamp": _now_iso(),
                "snapshot_id": snapshot_id,
                "snapshot_path": str(snapshot_path),
                "triggered_by_path": triggered_by_path,
            })


# ── Convenience for assembling tool lists ────────────────────────────────────


def filesystem_tool_schemas(*, include_write: bool) -> list[dict]:
    """
    Return the filesystem tool schemas for an agent.

    Agent A gets all three; Agent B only gets read_file and list_files
    (no write_file).
    """
    schemas = [READ_FILE_SCHEMA, LIST_FILES_SCHEMA]
    if include_write:
        schemas.append(WRITE_FILE_SCHEMA)
    return schemas

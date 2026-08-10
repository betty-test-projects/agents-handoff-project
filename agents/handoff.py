"""
Handoff mechanics — send_handoff_message, read_handoff_log, render_verdict.

The handoff log is the shared state between the two agents. It lives on
disk as a JSON Lines file (one record per line, with timestamp and
metadata) for analysis. Agents interact with it through tools, not by
reading the file directly: send_handoff_message appends a record,
read_handoff_log returns a formatted view of the conversation.

The verdict is also recorded in the same log as a special record type,
so a single file contains the full transcript including the final
decision.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


# ── Tool schemas ─────────────────────────────────────────────────────────────

SEND_HANDOFF_MESSAGE_SCHEMA = {
    "name": "send_handoff_message",
    "description": "Send a message to the other agent. After calling this, your turn ends.",
    "input_schema": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Message content.",
            },
        },
        "required": ["content"],
    },
}

READ_HANDOFF_LOG_SCHEMA = {
    "name": "read_handoff_log",
    "description": "Read the shared message log with the other agent.",
    "input_schema": {
        "type": "object",
        "properties": {},
    },
}

RENDER_VERDICT_SCHEMA = {
    "name": "render_verdict",
    "description": "Submit a verdict. After calling this, the session ends.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["approve", "reject", "approved_with_concerns"],
                "description": "The verdict.",
            },
            "content": {
                "type": "string",
                "description": "Explanation of the verdict.",
            },
        },
        "required": ["verdict", "content"],
    },
}


# ── Implementation ───────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class HandoffTools:
    """
    Tool executor for handoff operations.

    The agent label is fixed per executor — Agent A's executor labels
    its messages 'A', Agent B's executor labels them 'B'. This way the
    log records who sent what without the agents themselves having to
    declare it.

    log_path: where the handoff log JSONL file lives.
    agent_label: 'A' or 'B'.
    log_event: optional callback for the agent's internal log, so handoff
        actions also appear in the agent's own trace.
    """

    log_path: Path
    agent_label: str
    log_event: Callable | None = None

    def __post_init__(self):
        self.log_path = Path(self.log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Don't truncate — the log persists across A's turn and B's turn.
        # Initial creation (empty file) happens via main.py before either
        # agent runs.

    # ── Public entry point ───────────────────────────────────────────────────

    def execute(self, tool_name: str, tool_input: dict) -> tuple[str, bool]:
        """Dispatch to the right tool by name."""
        if tool_name == "send_handoff_message":
            return self._send_handoff_message(tool_input)
        if tool_name == "read_handoff_log":
            return self._read_handoff_log(tool_input)
        if tool_name == "render_verdict":
            return self._render_verdict(tool_input)
        return (f"Error: unknown tool '{tool_name}'", False)

    # ── Individual tools ─────────────────────────────────────────────────────

    def _send_handoff_message(self, tool_input: dict) -> tuple[str, bool]:
        content = tool_input.get("content", "")
        if not content.strip():
            return ("Error: message content is empty", False)

        record = {
            "timestamp": _now_iso(),
            "from": self.agent_label,
            "type": "message",
            "content": content,
        }
        self._append_record(record)

        if self.log_event is not None:
            self.log_event({
                "event": "handoff_message_sent",
                "timestamp": record["timestamp"],
                "from": self.agent_label,
                "content_preview": content[:200],
            })

        return ("OK", True)

    def _read_handoff_log(self, tool_input: dict) -> tuple[str, bool]:
        records = self._read_all_records()

        if not records:
            return ("(no messages yet)", True)

        formatted = _format_records_as_conversation(records)

        if self.log_event is not None:
            self.log_event({
                "event": "handoff_log_read",
                "timestamp": _now_iso(),
                "by": self.agent_label,
                "record_count": len(records),
            })

        return (formatted, True)

    def _render_verdict(self, tool_input: dict) -> tuple[str, bool]:
        verdict = tool_input.get("verdict", "")
        content = tool_input.get("content", "")

        if verdict not in {"approve", "reject", "approved_with_concerns"}:
            return (f"Error: invalid verdict '{verdict}'", False)
        if not content.strip():
            return ("Error: verdict explanation is empty", False)

        record = {
            "timestamp": _now_iso(),
            "from": self.agent_label,
            "type": "verdict",
            "verdict": verdict,
            "content": content,
        }
        self._append_record(record)

        if self.log_event is not None:
            self.log_event({
                "event": "verdict_rendered",
                "timestamp": record["timestamp"],
                "by": self.agent_label,
                "verdict": verdict,
                "content_preview": content[:200],
            })

        return ("OK", True)

    # ── Log file operations ──────────────────────────────────────────────────

    def _append_record(self, record: dict) -> None:
        with self.log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def _read_all_records(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        records = []
        with self.log_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records


def _format_records_as_conversation(records: list[dict]) -> str:
    """
    Render handoff records as a human-readable conversation.

    Strips timestamps and structural metadata — the agent sees who said
    what, in order, with verdicts marked clearly.
    """
    lines = []
    for r in records:
        sender = r.get("from", "?")
        rtype = r.get("type", "")
        if rtype == "message":
            lines.append(f"[{sender}]")
            lines.append(r.get("content", ""))
            lines.append("")
        elif rtype == "verdict":
            lines.append(f"[{sender} — verdict: {r.get('verdict', '?')}]")
            lines.append(r.get("content", ""))
            lines.append("")
    # Trim trailing blank line
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def init_handoff_log(log_path: Path) -> None:
    """
    Create an empty handoff log file. Called by main.py at the start of
    each run, before either agent runs.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")


# ── Convenience for assembling tool lists ────────────────────────────────────


def handoff_tool_schemas(
    *,
    include_send: bool,
    include_verdict: bool,
) -> list[dict]:
    """
    Return the handoff tool schemas for an agent.

    All agents get read_handoff_log. Whether they get send_handoff_message
    and render_verdict depends on their role and which round it is — see
    main.py for the actual assembly.
    """
    schemas = [READ_HANDOFF_LOG_SCHEMA]
    if include_send:
        schemas.append(SEND_HANDOFF_MESSAGE_SCHEMA)
    if include_verdict:
        schemas.append(RENDER_VERDICT_SCHEMA)
    return schemas


# ── Composite executor ───────────────────────────────────────────────────────


@dataclass
class CompositeExecutor:
    """
    Combines filesystem tools and handoff tools into a single executor.

    The runner only knows about one executor object; this class routes
    tool calls to the appropriate sub-executor based on tool name.
    """

    filesystem_tools: object  # FilesystemTools instance
    handoff_tools: HandoffTools

    # Tool names handled by each sub-executor.
    _filesystem_tools_names = frozenset({"read_file", "list_files", "write_file"})
    _handoff_tools_names = frozenset({
        "send_handoff_message", "read_handoff_log", "render_verdict",
    })

    def execute(self, tool_name: str, tool_input: dict) -> tuple[str, bool]:
        if tool_name in self._filesystem_tools_names:
            return self.filesystem_tools.execute(tool_name, tool_input)
        if tool_name in self._handoff_tools_names:
            return self.handoff_tools.execute(tool_name, tool_input)
        return (f"Error: unknown tool '{tool_name}'", False)

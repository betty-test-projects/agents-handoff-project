"""
Agent runner — single agent loop.

Given a system prompt, an initial user message, a set of tools, and a tool
executor, run a Claude tool-use loop until one of the terminal tools is
called or the turn limit is reached.

This module is intentionally agnostic about what the tools actually do.
Tool implementations live elsewhere (see tools.py and handoff.py); the
runner only knows how to dispatch tool calls to an executor and feed
results back into the conversation.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

import anthropic


MODEL = "claude-sonnet-4-6"
MAX_OUTPUT_TOKENS = 8192


class ToolExecutor(Protocol):
    """
    Interface for executing tool calls.

    Implementations are responsible for the actual side effects (reading
    files, writing files, appending to the handoff log, etc).
    """

    def execute(self, tool_name: str, tool_input: dict) -> tuple[str, bool]:
        """
        Execute a tool call.

        Returns:
            (result_text, success) — result_text is what will be sent back
            to the model as the tool_result content; success is whether the
            tool executed without an internal error.
        """
        ...


@dataclass
class RunResult:
    """Outcome of a single agent run."""

    status: str  # "completed" | "terminated_by_limit" | "error"
    terminal_tool: str | None = None
    terminal_tool_input: dict | None = None
    turn_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    error_message: str | None = None


@dataclass
class LogWriter:
    """
    Append-only JSON Lines writer for an agent's internal log.

    By default, creating a LogWriter truncates any existing file at the
    path — this matches the common case where each run starts fresh. Pass
    truncate=False to reuse an existing log across multiple rounds (the
    same agent running multiple turns).
    """

    path: Path
    truncate: bool = True

    def __post_init__(self):
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.truncate:
            self.path.write_text("")

    def write(self, event: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(event, default=str) + "\n")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_text(content_blocks: list) -> str:
    """
    Pull text content out of the model's response, ignoring tool_use blocks.
    """
    parts = []
    for block in content_blocks:
        if block.type == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()


def _extract_tool_uses(content_blocks: list) -> list[dict]:
    """Pull tool_use blocks out of the model's response."""
    tool_uses = []
    for block in content_blocks:
        if block.type == "tool_use":
            tool_uses.append({
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })
    return tool_uses


def run_agent(
    *,
    agent_label: str,
    system_prompt: str,
    initial_user_message: str,
    tools: list[dict],
    tool_executor: ToolExecutor,
    terminal_tool_names: set[str],
    max_turns: int,
    log_writer: LogWriter,
    round_num: int = 1,
    client: anthropic.Anthropic | None = None,
) -> RunResult:
    """
    Run a single agent loop.

    Args:
        agent_label: Identifier for logs (e.g. "A" or "B").
        system_prompt: Static system prompt for this agent.
        initial_user_message: First user message kicking off the loop.
        tools: List of tool schemas to expose to the model.
        tool_executor: Object that knows how to execute each tool by name.
        terminal_tool_names: Set of tool names that, when called, end the run.
        max_turns: Maximum number of API calls before forced termination.
        log_writer: LogWriter for this agent's internal log.
        round_num: Which round this is (recorded in turn entries).
        client: Optional pre-built Anthropic client (default: create one).
    """
    if client is None:
        client = anthropic.Anthropic()

    log = log_writer
    log.write({
        "event": "run_start",
        "agent": agent_label,
        "round": round_num,
        "timestamp": _now_iso(),
        "model": MODEL,
        "system_prompt": system_prompt,
        "initial_user_message": initial_user_message,
        "tools": [t["name"] for t in tools],
        "terminal_tool_names": sorted(terminal_tool_names),
        "max_turns": max_turns,
    })

    messages: list[dict] = [
        {"role": "user", "content": initial_user_message},
    ]

    total_input_tokens = 0
    total_output_tokens = 0
    terminal_tool: str | None = None
    terminal_tool_input: dict | None = None
    status = "turn_limit_reached"
    turn = 0

    for turn in range(1, max_turns + 1):
        try:
            response = _call_with_retry(
                client=client,
                system=system_prompt,
                messages=messages,
                tools=tools,
            )
        except Exception as e:
            log.write({
                "event": "error",
                "agent": agent_label,
                "round": round_num,
                "turn": turn,
                "timestamp": _now_iso(),
                "message": str(e),
            })
            log.write({
                "event": "run_end",
                "agent": agent_label,
                "round": round_num,
                "timestamp": _now_iso(),
                "status": "error",
                "total_turns": turn - 1,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "error_message": str(e),
            })
            return RunResult(
                status="error",
                turn_count=turn - 1,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                error_message=str(e),
            )

        turn_input_tokens = response.usage.input_tokens
        turn_output_tokens = response.usage.output_tokens
        total_input_tokens += turn_input_tokens
        total_output_tokens += turn_output_tokens

        model_said = _extract_text(response.content)
        tool_uses = _extract_tool_uses(response.content)

        tool_results = []
        terminal_hit = False
        for tu in tool_uses:
            if tu["name"] in terminal_tool_names:
                result_text, success = tool_executor.execute(
                    tu["name"], tu["input"]
                )
                tool_results.append({
                    "tool_use_id": tu["id"],
                    "name": tu["name"],
                    "input": tu["input"],
                    "result": result_text,
                    "success": success,
                    "terminal": True,
                })
                terminal_tool = tu["name"]
                terminal_tool_input = tu["input"]
                terminal_hit = True
                break
            else:
                result_text, success = tool_executor.execute(
                    tu["name"], tu["input"]
                )
                tool_results.append({
                    "tool_use_id": tu["id"],
                    "name": tu["name"],
                    "input": tu["input"],
                    "result": result_text,
                    "success": success,
                    "terminal": False,
                })

        log.write({
            "event": "turn",
            "agent": agent_label,
            "round": round_num,
            "turn": turn,
            "timestamp": _now_iso(),
            "model_said": model_said,
            "tool_calls": [
                {"id": tu["id"], "name": tu["name"], "input": tu["input"]}
                for tu in tool_uses
            ],
            "tool_results": tool_results,
            "stop_reason": response.stop_reason,
            "usage": {
                "input_tokens": turn_input_tokens,
                "output_tokens": turn_output_tokens,
            },
        })

        messages.append({
            "role": "assistant",
            "content": response.content,
        })

        if tool_results:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tr["tool_use_id"],
                        "content": tr["result"],
                    }
                    for tr in tool_results
                ],
            })

        if terminal_hit:
            status = "completed"
            break

        if response.stop_reason == "end_turn" and not tool_uses:
            status = "no_terminal_call"
            break

    log.write({
        "event": "run_end",
        "agent": agent_label,
        "round": round_num,
        "timestamp": _now_iso(),
        "status": status,
        "terminal_tool": terminal_tool,
        "terminal_tool_input": terminal_tool_input,
        "total_turns": turn,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    })

    return RunResult(
        status=status,
        terminal_tool=terminal_tool,
        terminal_tool_input=terminal_tool_input,
        turn_count=turn,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
    )


def _call_with_retry(
    *,
    client: anthropic.Anthropic,
    system: str,
    messages: list[dict],
    tools: list[dict],
    max_attempts: int = 2,
):
    """Call the API with one retry on transient errors."""
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            return client.messages.create(
                model=MODEL,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=system,
                messages=messages,
                tools=tools,
            )
        except (anthropic.APIConnectionError, anthropic.InternalServerError) as e:
            last_err = e
            if attempt < max_attempts:
                time.sleep(2)
                continue
            raise
        except Exception:
            raise
    raise last_err  # pragma: no cover

"""
main.py — orchestrator for a single agents-handoff run.

Usage:
    python main.py --spec brief
    python main.py --spec detailed --max-rounds 2
    python main.py --spec detailed --run-id pilot_test_1

What this does:
    1. Creates a fresh run directory under runs/<run_id>/
    2. Resets task_manager/ to the baseline copy
    3. Initializes an empty handoff log
    4. Runs the round loop: A -> B -> A -> B (up to max_rounds)
    5. On each round, constructs Agent A's and Agent B's tool lists,
       prompts, and executors
    6. Stops when B calls render_verdict or any agent fails
    7. Writes run_summary.json and prints a summary to stdout
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Make the agents package importable when running as a script.
sys.path.insert(0, str(Path(__file__).parent))

from agents.runner import LogWriter, run_agent
from agents.tools import FilesystemTools, filesystem_tool_schemas
from agents.handoff import (
    CompositeExecutor,
    HandoffTools,
    handoff_tool_schemas,
    init_handoff_log,
)


# ── Constants ────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent
TASK_MANAGER_DIR = REPO_ROOT / "task_manager"
TASK_MANAGER_BASELINE_DIR = REPO_ROOT / "task_manager_baseline"
SPECS_DIR = REPO_ROOT / "specs"
RUNS_DIR = REPO_ROOT / "runs"

MAX_TURNS_A = 20
MAX_TURNS_B = 15

# Pricing for claude-sonnet-4-6 (USD per million tokens).
INPUT_PRICE_PER_M = 3.0
OUTPUT_PRICE_PER_M = 15.0


# ── Prompt templates ─────────────────────────────────────────────────────────


def build_prompt_a(spec_path_for_agent: str, *, invite: bool = False) -> str:
    """
    System prompt for Agent A. Position language, no role labels.

    When `invite=True`, an explicit invitation is appended encouraging A to
    include open questions in the handoff message. This is the controlled
    variant for testing whether dialogue can be driven from A's side (rather
    than from B's tool list). See README "Two prompt modes" for narrative
    context.
    """
    invitation_text = (
        "\nYour handoff message may include questions for the other agent "
        "that would help their review.\n"
        if invite
        else ""
    )
    return (
        "You are working on a software project. The project root contains "
        "code files and a specification document.\n"
        "\n"
        "You have these tools:\n"
        "- read_file(path)\n"
        "- list_files(path)\n"
        "- write_file(path, content)\n"
        "- read_handoff_log()\n"
        "- send_handoff_message(content)\n"
        "\n"
        "A message log is shared with another agent. Use read_handoff_log "
        "to read it. After your turn ends, the other agent will read this log.\n"
        "\n"
        "Your turn ends when you call send_handoff_message. You may call "
        "other tools as many times as you need before then.\n"
        f"{invitation_text}"
        "\n"
        f"The specification is at {spec_path_for_agent}."
    )


def build_prompt_b(
    spec_path_for_agent: str,
    *,
    final_round: bool,
    invite: bool = False,
) -> str:
    """
    System prompt for Agent B.

    In non-final rounds, B has both send_handoff_message and render_verdict.
    In the final round, send_handoff_message is removed from the tool list
    (handled in main, not here) — the prompt itself does not change.

    When `invite=True` and round is non-final, an explicit invitation to use
    send_handoff_message is appended. This is the controlled variant for
    testing whether affordance requires explicit prompt-level invitation.
    See README "Two prompt modes" for narrative context.
    """
    if final_round:
        ending_options = (
            "Your turn ends when you call render_verdict, which ends the session."
        )
        # In final round, send_handoff_message is not in the tool list, so
        # the invitation is structurally moot and is omitted regardless of `invite`.
        invitation_text = ""
    else:
        ending_options = (
            "Your turn ends when you call send_handoff_message or render_verdict.\n"
            "- send_handoff_message: the other agent may respond, then your turn resumes.\n"
            "- render_verdict: the session ends."
        )
        invitation_text = (
            "\nYou may use send_handoff_message to ask the other agent anything "
            "that would help your verdict.\n"
            if invite
            else ""
        )

    return (
        "You are working on a software project. The project root contains "
        "code files and a specification document.\n"
        "\n"
        "You have these tools:\n"
        + _b_tool_list_text(final_round=final_round)
        + "\n\n"
        "A message log is shared with another agent. Another agent has "
        "already written to this log before your turn. Use read_handoff_log "
        "to read it.\n"
        "\n"
        f"{ending_options}\n"
        f"{invitation_text}"
        "\n"
        f"The specification is at {spec_path_for_agent}."
    )


def _b_tool_list_text(*, final_round: bool) -> str:
    lines = [
        "- read_file(path)",
        "- list_files(path)",
        "- read_handoff_log()",
    ]
    if not final_round:
        lines.append("- send_handoff_message(content)")
    lines.append("- render_verdict(verdict, content)")
    return "\n".join(lines)


# ── Initial user messages ────────────────────────────────────────────────────


def initial_user_message_a() -> str:
    """First user message kicking off Agent A's loop."""
    return "Begin."


def initial_user_message_b() -> str:
    """First user message kicking off Agent B's loop."""
    return "Begin."


# Note: we use minimal kickoff messages. The system prompt has already
# described the world; the user message just signals "go". This avoids
# duplicating instructions or injecting task framing.


# ── Setup helpers ────────────────────────────────────────────────────────────


def ensure_baseline_exists() -> None:
    """
    Ensure task_manager_baseline/ exists. On first run, create it by
    copying the current task_manager/ directory.
    """
    if TASK_MANAGER_BASELINE_DIR.exists():
        return
    if not TASK_MANAGER_DIR.exists():
        raise RuntimeError(
            f"Neither {TASK_MANAGER_BASELINE_DIR} nor {TASK_MANAGER_DIR} "
            "exists. Cannot bootstrap the experiment."
        )
    print(
        f"[setup] Creating baseline copy at {TASK_MANAGER_BASELINE_DIR}",
        file=sys.stderr,
    )
    shutil.copytree(
        TASK_MANAGER_DIR,
        TASK_MANAGER_BASELINE_DIR,
        ignore=shutil.ignore_patterns(
            "*.db", "*.db-journal", "__pycache__", "*.pyc", ".DS_Store"
        ),
    )


def reset_task_manager() -> None:
    """Reset task_manager/ to the baseline state."""
    if TASK_MANAGER_DIR.exists():
        shutil.rmtree(TASK_MANAGER_DIR)
    shutil.copytree(TASK_MANAGER_BASELINE_DIR, TASK_MANAGER_DIR)


def generate_run_id(spec: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{spec}"


def resolve_spec_path(spec: str) -> tuple[Path, str]:
    """
    Map the spec short name to its file path.

    Returns:
        (absolute_path, path_as_seen_by_agent) — the agent sees the path
        as 'specs/recurring_tasks_<spec>.md' (the spec lives outside the
        sandbox, but the agent's prompt uses this path for reference;
        agents access the spec via read_file which is sandboxed to
        task_manager — so we copy the spec into task_manager too).
    """
    filename = f"recurring_tasks_{spec}.md"
    abs_path = SPECS_DIR / filename
    if not abs_path.exists():
        raise FileNotFoundError(f"Spec file not found: {abs_path}")
    # The agent will see the spec as a file inside its sandbox.
    return abs_path, filename


def stage_spec_into_sandbox(spec_abs_path: Path) -> str:
    """
    Copy the chosen spec into task_manager/ so the agent can read it
    via its sandboxed read_file. Returns the agent-visible filename.
    """
    target_name = spec_abs_path.name  # e.g. recurring_tasks_brief.md
    shutil.copy2(spec_abs_path, TASK_MANAGER_DIR / target_name)
    return target_name


# ── Cost estimation ──────────────────────────────────────────────────────────


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens / 1_000_000 * INPUT_PRICE_PER_M
        + output_tokens / 1_000_000 * OUTPUT_PRICE_PER_M
    )


# ── Round orchestration ──────────────────────────────────────────────────────


def run_session(
    *,
    spec: str,
    max_rounds: int,
    run_id: str,
    invite_a: bool,
    invite_b: bool,
) -> dict:
    """
    Execute a full session: setup, round loop, summary.

    Returns the run summary dict.

    `invite_a` controls whether Agent A's prompt receives an explicit
    invitation to include open questions in the handoff message.
    `invite_b` controls whether Agent B's prompt receives an explicit
    invitation to use send_handoff_message before rendering verdict.
    See README "Two prompt modes" for narrative context.
    """
    # Resolve paths
    spec_abs_path, _ = resolve_spec_path(spec)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    handoff_log_path = run_dir / "handoff_log.jsonl"
    a_internal_log = run_dir / "agent_a_internal.jsonl"
    b_internal_log = run_dir / "agent_b_internal.jsonl"
    snapshots_dir = run_dir / "snapshots"

    # Setup
    ensure_baseline_exists()
    reset_task_manager()
    agent_visible_spec_name = stage_spec_into_sandbox(spec_abs_path)
    init_handoff_log(handoff_log_path)

    print(f"[run] run_id={run_id}")
    print(f"[run] spec={spec} ({agent_visible_spec_name})")
    print(f"[run] max_rounds={max_rounds}")
    print(f"[run] run_dir={run_dir}")
    print()

    # Persistent log writers (truncate=True on first creation, then we
    # reuse the same writer object across rounds so all entries land
    # in the same file).
    a_log = LogWriter(a_internal_log, truncate=True)
    b_log = LogWriter(b_internal_log, truncate=True)

    # Round loop
    started_at = datetime.now(timezone.utc).isoformat()
    final_verdict: dict | None = None
    failure_reason: str | None = None

    a_input_total = 0
    a_output_total = 0
    b_input_total = 0
    b_output_total = 0
    a_turns_total = 0
    b_turns_total = 0

    for round_num in range(1, max_rounds + 1):
        is_final_round = (round_num == max_rounds)
        print(f"[round {round_num}/{max_rounds}] starting")

        # ── Agent A ──
        print(f"[round {round_num}] A running...")
        a_result = _run_agent_a(
            round_num=round_num,
            spec_name_for_agent=agent_visible_spec_name,
            handoff_log_path=handoff_log_path,
            snapshots_dir=snapshots_dir,
            log_writer=a_log,
            invite=invite_a,
        )
        a_input_total += a_result.input_tokens
        a_output_total += a_result.output_tokens
        a_turns_total += a_result.turn_count
        print(
            f"[round {round_num}] A done: status={a_result.status} "
            f"turns={a_result.turn_count} "
            f"terminal_tool={a_result.terminal_tool}"
        )

        if a_result.status != "completed":
            failure_reason = (
                f"Agent A failed in round {round_num}: {a_result.status}"
                + (f" ({a_result.error_message})" if a_result.error_message else "")
            )
            break

        # ── Agent B ──
        print(f"[round {round_num}] B running...")
        b_result = _run_agent_b(
            round_num=round_num,
            is_final_round=is_final_round,
            spec_name_for_agent=agent_visible_spec_name,
            handoff_log_path=handoff_log_path,
            snapshots_dir=snapshots_dir,
            log_writer=b_log,
            invite=invite_b,
        )
        b_input_total += b_result.input_tokens
        b_output_total += b_result.output_tokens
        b_turns_total += b_result.turn_count
        print(
            f"[round {round_num}] B done: status={b_result.status} "
            f"turns={b_result.turn_count} "
            f"terminal_tool={b_result.terminal_tool}"
        )

        if b_result.status != "completed":
            failure_reason = (
                f"Agent B failed in round {round_num}: {b_result.status}"
                + (f" ({b_result.error_message})" if b_result.error_message else "")
            )
            break

        if b_result.terminal_tool == "render_verdict":
            final_verdict = {
                "verdict": b_result.terminal_tool_input.get("verdict"),
                "content": b_result.terminal_tool_input.get("content"),
            }
            print(
                f"[round {round_num}] verdict rendered: {final_verdict['verdict']}"
            )
            break

        # Otherwise B sent a handoff message — continue to next round.
        print(f"[round {round_num}] B sent message, continuing to next round")
        print()

    ended_at = datetime.now(timezone.utc).isoformat()

    # ── Summary ──
    total_input = a_input_total + b_input_total
    total_output = a_output_total + b_output_total
    summary = {
        "run_id": run_id,
        "spec": spec,
        "max_rounds": max_rounds,
        "a_invite_enabled": invite_a,
        "b_invite_enabled": invite_b,
        "started_at": started_at,
        "ended_at": ended_at,
        "agent_a": {
            "total_turns": a_turns_total,
            "input_tokens": a_input_total,
            "output_tokens": a_output_total,
        },
        "agent_b": {
            "total_turns": b_turns_total,
            "input_tokens": b_input_total,
            "output_tokens": b_output_total,
        },
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "estimated_cost_usd": round(estimate_cost_usd(total_input, total_output), 4),
        "final_verdict": final_verdict,
        "failure_reason": failure_reason,
        "status": "failed" if failure_reason else ("completed" if final_verdict else "no_verdict"),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Print summary to stdout
    print()
    print("=" * 60)
    print("Run summary")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    return summary


def _run_agent_a(
    *,
    round_num: int,
    spec_name_for_agent: str,
    handoff_log_path: Path,
    snapshots_dir: Path,
    log_writer: LogWriter,
    invite: bool,
):
    """Construct Agent A's executor + tools and run."""
    fs_tools = FilesystemTools(
        project_root=TASK_MANAGER_DIR,
        snapshots_dir=snapshots_dir,
        log_event=log_writer.write,
    )
    handoff = HandoffTools(
        log_path=handoff_log_path,
        agent_label="A",
        log_event=log_writer.write,
    )
    executor = CompositeExecutor(filesystem_tools=fs_tools, handoff_tools=handoff)

    tools = (
        filesystem_tool_schemas(include_write=True)
        + handoff_tool_schemas(include_send=True, include_verdict=False)
    )

    return run_agent(
        agent_label="A",
        system_prompt=build_prompt_a(spec_name_for_agent, invite=invite),
        initial_user_message=initial_user_message_a(),
        tools=tools,
        tool_executor=executor,
        terminal_tool_names={"send_handoff_message"},
        max_turns=MAX_TURNS_A,
        log_writer=log_writer,
        round_num=round_num,
    )


def _run_agent_b(
    *,
    round_num: int,
    is_final_round: bool,
    spec_name_for_agent: str,
    handoff_log_path: Path,
    snapshots_dir: Path,
    log_writer: LogWriter,
    invite: bool,
):
    """Construct Agent B's executor + tools and run."""
    fs_tools = FilesystemTools(
        project_root=TASK_MANAGER_DIR,
        snapshots_dir=snapshots_dir,
        log_event=log_writer.write,
    )
    handoff = HandoffTools(
        log_path=handoff_log_path,
        agent_label="B",
        log_event=log_writer.write,
    )
    executor = CompositeExecutor(filesystem_tools=fs_tools, handoff_tools=handoff)

    # B never has write_file. In final round, no send_handoff_message either.
    tools = (
        filesystem_tool_schemas(include_write=False)
        + handoff_tool_schemas(
            include_send=not is_final_round,
            include_verdict=True,
        )
    )

    terminal_tool_names = {"render_verdict"}
    if not is_final_round:
        terminal_tool_names.add("send_handoff_message")

    return run_agent(
        agent_label="B",
        system_prompt=build_prompt_b(
            spec_name_for_agent,
            final_round=is_final_round,
            invite=invite,
        ),
        initial_user_message=initial_user_message_b(),
        tools=tools,
        tool_executor=executor,
        terminal_tool_names=terminal_tool_names,
        max_turns=MAX_TURNS_B,
        log_writer=log_writer,
        round_num=round_num,
    )


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> int:
    load_dotenv()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 2

    parser = argparse.ArgumentParser(description="Run one agents-handoff session.")
    parser.add_argument(
        "--spec",
        required=True,
        choices=["brief", "detailed"],
        help="Which spec to give the agents",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=2,
        help="Maximum number of A->B rounds (default: 2)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Custom run id (default: auto-generated from timestamp)",
    )
    parser.add_argument(
        "--invite-a",
        action="store_true",
        dest="invite_a",
        help="Append explicit invitation in Agent A's prompt to include "
             "open questions in the handoff message (controlled variant; "
             "see README 'Two prompt modes').",
    )
    parser.add_argument(
        "--invite-b",
        "--invite",  # deprecated alias for backward compatibility
        action="store_true",
        dest="invite_b",
        help="Append explicit invitation in Agent B's prompt to use "
             "send_handoff_message before rendering verdict (controlled "
             "variant; see README 'Two prompt modes'). `--invite` is a "
             "deprecated alias.",
    )
    args = parser.parse_args()

    run_id = args.run_id or generate_run_id(args.spec)

    try:
        summary = run_session(
            spec=args.spec,
            max_rounds=args.max_rounds,
            run_id=run_id,
            invite_a=args.invite_a,
            invite_b=args.invite_b,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if summary["status"] == "failed":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

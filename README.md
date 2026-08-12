# When Agents Talk

An experimental apparatus for observing what happens when two Claude agents must coordinate through a shared message log. Companion code to the five-part article series *When Agents Talk*.

## Research question

Prior work observed a single agent making autonomous decisions inside one task. This series extends that observation into a two-agent handoff: one agent implements a feature, the other reviews it, and they must communicate through a shared channel to converge on a verdict.

The question is what distributed autonomy looks like — where the joints hold, where they slip, and what gets ratified between two agents that never talk to the user.

The task substrate is a small Flask task manager. The feature to implement is recurring tasks. The spec is provided in two versions (brief and detailed) so the same handoff can be observed under different amounts of specification pressure.

## Design

### Two agents, asymmetric tools

Agent A can read the codebase, list files, and write files. Agent A can send messages through the handoff log but cannot render a verdict.

Agent B can read the codebase and list files but not write. Agent B can send messages through the handoff log and can render a verdict — one of `approve`, `reject`, or `approved_with_concerns`. A verdict ends the session.

Both agents see the same spec. Information is symmetric; only capabilities differ. Any tension in the conversation comes from that asymmetry, not from hidden information.

### Handoff channel

A single JSON Lines file, written to and read from through tool calls. Neither agent reads the file directly — `send_handoff_message` appends, `read_handoff_log` returns a formatted view of prior messages, `render_verdict` records the decision in the same log.

The channel is bidirectional. A round is one turn from A followed by one turn from B. `max_rounds` is capped at 2 by default. B can either message A back (extending the conversation) or render a verdict (ending the session).

### Prompt strategy

Position language only, no role framing. Neither prompt names the agent as "developer", "engineer", "reviewer", or "QA". The system prompts describe the tools available, the shape of the message log, and the location of the spec. The verdict enum is stated but not motivated. Whatever role-appropriate behavior emerges, emerges from the task shape.

## Findings

Each item below is a short pointer to the finding and the article that develops it.

**Turn shape held across every baseline configuration.** Four baseline runs produced identical turn counts (A = 8, B = 5), B never called `send_handoff_message`, and the verdict tracked spec verbosity — brief specs produced `approve`, detailed specs produced `approved_with_concerns`. → [Part 3 — Where Nobody Asked Back](https://medium.com/@betty.lin.twn/when-agents-talk-part-3-where-nobody-asked-back-749334d48393)

**Invitation must originate from the sender side, not the receiver.** Appending an explicit invitation to Agent B's prompt changed nothing observable. Appending an equivalent invitation to Agent A's prompt triggered the first two-round dialogue in the entire experiment. → [Part 4 — Inviting the Wrong Side First](https://medium.com/@betty.lin.twn/when-agents-talk-part-4-the-conversation-begins-a6d0fbfac07c)

**Dialogue between agents converges on the spec, not the user.** When dialogue did emerge, A and B bilaterally ratified a spec-consistent outcome that broke the user experience — newly created recurring tasks visibly disappeared from the task list, leaving only a small toast notification behind. → [Part 4 — Inviting the Wrong Side First](https://medium.com/@betty.lin.twn/when-agents-talk-part-4-the-conversation-begins-a6d0fbfac07c)

**QA begins where the specification ends.** Agent B did not perform QA verification in any run. Its failures were noticing-level, not surfacing-level: the model did not intelligently fill gaps left by spec vagueness. Useful output from vague prompts, when it appears, reflects learned user patterns, coincidence, or base rate. → [Part 5 — The Role That Stayed With Me](https://medium.com/@betty.lin.twn/when-agents-talk-part-5-the-role-that-stayed-with-me-6c7875a4306f)

The series opens with [Part 1 — Two Agents, One Room](https://medium.com/@betty.lin.twn/when-agents-talk-part-1-two-agents-one-room-9fcd97e5eb3f) and [Part 2 — The Roles I Didn't Assign](https://medium.com/@betty.lin.twn/when-agents-talk-part-2-the-roles-i-didnt-assign-b6b766d5e435), which cover motivation and design; Parts 3–5 report observations.

## Reproducing the experiments

### Requirements

Python 3.13, an Anthropic API key, and the packages listed in `requirements.txt`. Set `ANTHROPIC_API_KEY` in a `.env` file at the repo root (see `.env.example`).

### Repo layout

```
agents-handoff-project/
├── main.py                       # session orchestrator
├── agents/
│   ├── runner.py                 # single-agent tool-use loop
│   ├── tools.py                  # sandboxed filesystem tools
│   └── handoff.py                # handoff channel and verdict tools
├── specs/
│   ├── recurring_tasks_brief.md
│   └── recurring_tasks_detailed.md
├── task_manager/                 # sandbox; reset before each run
│   ├── app.py
│   └── templates/
│       └── index.html
├── task_manager_baseline/        # auto-created on first run
└── runs/                         # auto-created; one folder per run
```

On first run, `main.py` creates `task_manager_baseline/` by copying the current state of `task_manager/`. Every subsequent run resets `task_manager/` from that baseline before either agent begins.

### Running a session

```bash
python main.py --spec brief
python main.py --spec detailed
python main.py --spec detailed --run-id my_custom_id
python main.py --spec detailed --max-rounds 2
```

Each invocation writes a new directory under `runs/`.

### What each run produces

```
runs/<run_id>/
├── run_summary.json           # verdict, turn counts, token usage, cost
├── handoff_log.jsonl          # shared message log (both agents visible)
├── agent_a_internal.jsonl     # A's full turn-by-turn trace
├── agent_b_internal.jsonl     # B's full turn-by-turn trace
└── snapshots/                 # code snapshot after every write_file
```

The internal logs record every model call, every tool call, and the model's text between calls. The handoff log records only what one agent chose to show the other.

### Experiment variants

Three configurations were run in this experiment. The variants differ only in one sentence appended to one system prompt in `main.py`.

**Baseline.** Prompts as they appear in `build_prompt_a` and `build_prompt_b`. Ran twice with `--spec brief` and twice with `--spec detailed`.

**invite-b.** Append to Agent B's system prompt, before the final line:

> *You may use send_handoff_message to ask the other agent anything that would help your verdict.*

Agent A's prompt unchanged. Result: no dialogue, no observable behavior change from baseline.

**invite-a.** Append to Agent A's system prompt, before the final line:

> *Your handoff message may include questions for the other agent that would help their review.*

Agent B's prompt unchanged. Result: the first two-round dialogue in the experiment, and the finding developed in Part 4.

## Related work

This work follows *When the Agent Chooses*, a prior three-part series on single-agent QA autonomy. Its entry premise — *prompt operates as a filter, not an enabler* — is treated here as given rather than re-argued.

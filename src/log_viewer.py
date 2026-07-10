"""
Log persistence + human-readable rendering.

persist_run_log()  -> writes a RunLog to logs/<run_id>.json
load_run_log()     -> reads it back
render_run_log_html() -> used both by the FastAPI /logs/{run_id} route and
                          can be dumped to a standalone .html file
print_run_log_table() -> CLI viewer using `rich`, for `python -m src.log_viewer <run_id>`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import settings
from .schemas import RunLog

console = Console()


def persist_run_log(run_log: RunLog) -> Path:
    path = settings.log_dir / f"{run_log.run_id}.json"
    path.write_text(run_log.model_dump_json(indent=2))
    return path


def load_run_log(run_id: str) -> RunLog:
    path = settings.log_dir / f"{run_id}.json"
    data = json.loads(path.read_text())
    return RunLog(**data)


def list_run_ids() -> list[str]:
    return sorted(p.stem for p in settings.log_dir.glob("*.json"))


# ---------------------------------------------------------------------------
# CLI (rich table) viewer
# ---------------------------------------------------------------------------

def print_run_log_table(run_log: RunLog) -> None:
    console.print(Panel.fit(
        f"[bold]Run {run_log.run_id}[/bold]\n"
        f"Goal: {run_log.goal}\n"
        f"Self-correction: {run_log.self_correction_enabled}\n"
        f"Completed: {run_log.completed}  |  Self-corrections: {run_log.self_corrections}  |  "
        f"Unresolved: {len(run_log.unresolved_subtasks)}  |  Duration: {run_log.duration_seconds()}s",
        title="Run Summary",
    ))

    table = Table(show_lines=True)
    table.add_column("Step", style="cyan", no_wrap=True)
    table.add_column("Thought", overflow="fold")
    table.add_column("Action", overflow="fold")
    table.add_column("Result", overflow="fold")
    table.add_column("Self-Eval", overflow="fold")
    table.add_column("Recovery", overflow="fold")

    for s in run_log.steps:
        action_str = f"{s.reasoning.action.tool_name}({s.reasoning.action.tool_input})" if s.reasoning.action else "-"
        result_str = "-"
        if s.tool_result:
            result_str = "OK" if s.tool_result.success else f"ERROR: {s.tool_result.error}"
        eval_str = "-"
        if s.self_evaluation:
            eval_str = f"[{s.self_evaluation.failure_mode.value}] {s.self_evaluation.rationale}"
        recovery_str = "-"
        if s.recovery:
            recovery_str = f"{s.recovery.strategy.value}: {s.recovery.rationale}"
        table.add_row(s.step_id, s.reasoning.thought, action_str, result_str, eval_str, recovery_str)

    console.print(table)
    if run_log.final_output:
        console.print(Panel(run_log.final_output, title="Final Output"))
    if run_log.unresolved_subtasks:
        console.print(Panel(", ".join(run_log.unresolved_subtasks), title="[red]Unresolved Subtasks[/red]"))


# ---------------------------------------------------------------------------
# HTML viewer (shared by CLI export and the FastAPI /logs route)
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Run {run_id}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; background:#0f1115; color:#e6e6e6; margin:0; padding:24px; }}
  h1 {{ font-size:18px; }}
  .meta {{ color:#9aa0a6; margin-bottom:20px; }}
  .step {{ border:1px solid #2a2d34; border-radius:10px; padding:14px 16px; margin-bottom:14px; background:#161923; }}
  .step-header {{ font-weight:600; color:#8ab4f8; margin-bottom:8px; }}
  .thought {{ color:#f4d35e; margin-bottom:6px; }}
  .action {{ font-family:monospace; background:#0b0d12; padding:6px 8px; border-radius:6px; margin-bottom:6px; white-space:pre-wrap; }}
  .observation {{ color:#c9c9c9; white-space:pre-wrap; margin-bottom:6px; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:10px; font-size:12px; margin-right:6px; }}
  .ok {{ background:#1e3a2a; color:#7ee787; }}
  .fail {{ background:#3a1e1e; color:#ff9b9b; }}
  .eval {{ background:#241f33; color:#c3a6ff; padding:6px 8px; border-radius:6px; margin-bottom:6px; }}
  .recovery {{ background:#332a1f; color:#ffcf86; padding:6px 8px; border-radius:6px; }}
  .final {{ border:1px solid #2a2d34; border-radius:10px; padding:16px; background:#12241a; margin-top:20px; }}
  .unresolved {{ border:1px solid #4a2020; border-radius:10px; padding:16px; background:#241212; margin-top:12px; }}
</style>
</head>
<body>
<h1>Run {run_id}</h1>
<div class="meta">
  Goal: {goal}<br>
  Self-correction enabled: {self_correction_enabled}<br>
  Completed: {completed} &nbsp;|&nbsp; Self-corrections: {self_corrections} &nbsp;|&nbsp;
  Unresolved: {n_unresolved} &nbsp;|&nbsp; Duration: {duration}s
</div>
{steps_html}
<div class="final"><b>Final Output</b><br>{final_output}</div>
{unresolved_html}
</body>
</html>
"""


def render_run_log_html(run_log: RunLog) -> str:
    steps_html = []
    for s in run_log.steps:
        action = s.reasoning.action
        action_str = f"{action.tool_name}({action.tool_input})" if action else "-"
        ok = s.tool_result.success if s.tool_result else False
        badge = f'<span class="badge {"ok" if ok else "fail"}">{"success" if ok else "failed"}</span>'
        eval_html = ""
        if s.self_evaluation:
            e = s.self_evaluation
            eval_html = (
                f'<div class="eval">self-eval: failure_mode={e.failure_mode.value}, '
                f'on_track={e.still_on_track}, confidence={e.confidence:.2f}<br>{e.rationale}</div>'
            )
        recovery_html = ""
        if s.recovery:
            r = s.recovery
            recovery_html = f'<div class="recovery">recovery: {r.strategy.value}<br>{r.rationale}</div>'
        obs = (s.tool_result.output if ok else s.tool_result.error) if s.tool_result else ""
        steps_html.append(f"""
        <div class="step">
          <div class="step-header">Step {s.step_id} {badge}</div>
          <div class="thought">💭 {s.reasoning.thought}</div>
          <div class="action">▶ {action_str}</div>
          <div class="observation">{obs}</div>
          {eval_html}
          {recovery_html}
        </div>""")

    unresolved_html = ""
    if run_log.unresolved_subtasks:
        unresolved_html = f'<div class="unresolved"><b>Unresolved Subtasks</b><br>{", ".join(run_log.unresolved_subtasks)}</div>'

    return _HTML_TEMPLATE.format(
        run_id=run_log.run_id,
        goal=run_log.goal,
        self_correction_enabled=run_log.self_correction_enabled,
        completed=run_log.completed,
        self_corrections=run_log.self_corrections,
        n_unresolved=len(run_log.unresolved_subtasks),
        duration=run_log.duration_seconds(),
        steps_html="".join(steps_html),
        final_output=run_log.final_output or "",
        unresolved_html=unresolved_html,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("Usage: python -m src.log_viewer <run_id>   (or 'latest' / 'all')")
        sys.exit(1)
    arg = sys.argv[1]
    ids = list_run_ids()
    if arg == "all":
        targets = ids
    elif arg == "latest":
        targets = ids[-1:]
    else:
        targets = [arg]
    for rid in targets:
        print_run_log_table(load_run_log(rid))

"""
Evaluation rig.

Runs the agent across 10 distinct code-generation goals, twice each:
  (a) self_correction=True   (the real system)
  (b) self_correction=False  (baseline / control condition)

For every run it persists a structured log to logs/<run_id>.json, then
prints + writes a comparison report (eval_report.md) covering: completion
rate, average steps taken, number of self-corrections triggered, and cases
where recovery failed (i.e. subtasks that ended up unresolved even with
self-correction on).

Usage:
    python run_evaluation.py            # uses LLM_MODE from .env
    python run_evaluation.py --mock     # force offline mock LLM
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

GOALS = [
    "Write a function that checks if a string is a palindrome, ignoring case and spaces, with tests.",
    "Write a function that computes the nth Fibonacci number iteratively, with tests for edge cases.",
    "Write a function that flattens an arbitrarily nested list, with tests.",
    "Write a function that merges two sorted lists into one sorted list, with tests.",
    "Write a function that counts word frequency in a string and returns the top 3 words, with tests.",
    "Write a function that validates whether a given string is balanced parentheses, with tests.",
    "Write a function that reverses the words in a sentence without reversing the letters, with tests.",
    "Write a function that finds the longest common prefix among a list of strings, with tests.",
    "Write a function that removes duplicate elements from a list while preserving order, with tests.",
    "Write a function that computes the factorial of a non-negative integer recursively, with tests.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="force LLM_MODE=mock for this run")
    parser.add_argument("--n", type=int, default=len(GOALS), help="number of goals to run (max 10)")
    args = parser.parse_args()

    if args.mock:
        os.environ["LLM_MODE"] = "mock"

    # import after potentially setting LLM_MODE, since config.py reads env at import time
    from src.agent import Agent
    from src.log_viewer import persist_run_log

    goals = GOALS[: args.n]
    agent = Agent()

    rows = []  # (goal, mode, run_log)
    for goal in goals:
        for self_correction in (True, False):
            t0 = time.time()
            run_log = agent.run(goal, self_correction=self_correction)
            elapsed = time.time() - t0
            path = persist_run_log(run_log)
            rows.append((goal, self_correction, run_log, elapsed, path))
            mode = "self-correct" if self_correction else "baseline"
            print(f"[{mode:12}] {goal[:55]:55} -> completed={run_log.completed} "
                  f"steps={len(run_log.steps)} corrections={run_log.self_corrections} "
                  f"unresolved={len(run_log.unresolved_subtasks)} ({elapsed:.1f}s) log={path.name}")

    _write_report(rows)


def _write_report(rows):
    sc_rows = [r for r in rows if r[1] is True]
    base_rows = [r for r in rows if r[1] is False]

    def summarize(rs):
        n = len(rs)
        completed = sum(1 for r in rs if r[2].completed)
        avg_steps = statistics.mean(len(r[2].steps) for r in rs) if rs else 0
        total_corrections = sum(r[2].self_corrections for r in rs)
        failed_recoveries = [
            (r[0], r[2].run_id, r[2].unresolved_subtasks) for r in rs if r[2].unresolved_subtasks
        ]
        return {
            "n": n,
            "completion_rate": completed / n if n else 0,
            "avg_steps": avg_steps,
            "total_corrections": total_corrections,
            "failed_recoveries": failed_recoveries,
        }

    sc_summary = summarize(sc_rows)
    base_summary = summarize(base_rows)

    lines = ["# Evaluation Report\n"]
    lines.append(f"Goals evaluated: {len(sc_rows)}\n")
    lines.append("## Self-correcting agent\n")
    lines.append(f"- Completion rate: {sc_summary['completion_rate']*100:.0f}%")
    lines.append(f"- Average steps taken: {sc_summary['avg_steps']:.1f}")
    lines.append(f"- Total self-corrections triggered: {sc_summary['total_corrections']}")
    lines.append(f"- Runs with unresolved (recovery-failed) subtasks: {len(sc_summary['failed_recoveries'])}")
    if sc_summary["failed_recoveries"]:
        lines.append("\n  Failed recoveries:")
        for goal, run_id, unresolved in sc_summary["failed_recoveries"]:
            lines.append(f"  - [{run_id}] {goal[:60]} -> unresolved: {unresolved}")

    lines.append("\n## Baseline (no self-correction)\n")
    lines.append(f"- Completion rate: {base_summary['completion_rate']*100:.0f}%")
    lines.append(f"- Average steps taken: {base_summary['avg_steps']:.1f}")
    lines.append(f"- Total self-corrections triggered: {base_summary['total_corrections']} (baseline never self-corrects, by design)")
    lines.append(f"- Runs with unresolved subtasks (tool just failed silently): {len(base_summary['failed_recoveries'])}")
    if base_summary["failed_recoveries"]:
        lines.append("\n  Unresolved (no recovery attempted):")
        for goal, run_id, unresolved in base_summary["failed_recoveries"]:
            lines.append(f"  - [{run_id}] {goal[:60]} -> unresolved: {unresolved}")

    lines.append("\n## Delta\n")
    delta_completion = (sc_summary["completion_rate"] - base_summary["completion_rate"]) * 100
    lines.append(f"- Completion rate improvement from self-correction: {delta_completion:+.0f} percentage points")

    report = "\n".join(lines)
    out_path = Path(__file__).resolve().parent / "eval_report.md"
    out_path.write_text(report)
    print("\n" + report)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

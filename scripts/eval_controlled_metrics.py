"""Evaluate controlled batch logs by method."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def load_method_logs(
    method_dir: Path,
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    logs = []
    for summary_path in sorted(method_dir.glob("*/episode_summary.json")):
        step_log_path = summary_path.parent / "step_log.jsonl"
        if not step_log_path.exists():
            continue
        logs.append((read_json(summary_path), read_jsonl(step_log_path)))
    return logs


def count_wrong_inspections(
    summary: dict[str, Any], step_logs: list[dict[str, Any]]
) -> int:
    wrong_instance_id = summary.get("wrong_instance_id")
    return sum(
        1
        for row in step_logs
        if row.get("selected_instance_id") == wrong_instance_id
        and float(row.get("evidence", 0.0)) == 0.0
    )


def switched_after_wrong(
    summary: dict[str, Any], step_logs: list[dict[str, Any]]
) -> bool:
    wrong_instance_id = summary.get("wrong_instance_id")
    if wrong_instance_id is None:
        return False

    saw_wrong = False
    for row in step_logs:
        selected_id = row.get("selected_instance_id")
        if selected_id == wrong_instance_id:
            saw_wrong = True
        elif saw_wrong:
            return True
    return False


def count_spf_triggers(step_logs: list[dict[str, Any]]) -> int:
    return sum(1 for row in step_logs if bool(row.get("spf_triggered", False)))


def get_final_reliability(
    summary: dict[str, Any],
    step_logs: list[dict[str, Any]],
    instance_id: str | None,
) -> float | None:
    if instance_id is None:
        return None

    final_memory = summary.get("final_memory") or []
    for instance in final_memory:
        if instance.get("instance_id") == instance_id:
            reliability = instance.get("reliability")
            return None if reliability is None else float(reliability)

    for row in reversed(step_logs):
        for candidate in row.get("top_candidate_scores", []) or []:
            if candidate.get("instance_id") == instance_id:
                reliability = candidate.get("reliability")
                return None if reliability is None else float(reliability)

    for row in reversed(step_logs):
        if row.get("selected_instance_id") == instance_id:
            reliability = row.get("reliability_after")
            return None if reliability is None else float(reliability)

    return None


def success_rate(summaries: list[dict[str, Any]]) -> float:
    if not summaries:
        return 0.0
    return mean([1.0 if summary.get("success", False) else 0.0 for summary in summaries])


def evaluate_method(
    method: str,
    method_logs: list[tuple[dict[str, Any], list[dict[str, Any]]]],
) -> dict[str, Any]:
    summaries = [summary for summary, _ in method_logs]
    normal_summaries = [
        summary
        for summary in summaries
        if summary.get("episode_type") == "normal-prior"
    ]
    misleading_summaries = [
        summary
        for summary in summaries
        if summary.get("episode_type") == "misleading-prior"
    ]
    misleading_logs = [
        (summary, step_logs)
        for summary, step_logs in method_logs
        if summary.get("episode_type") == "misleading-prior"
    ]

    reliability_drops = [
        float(summary["reliability_drop_wrong"])
        for summary in summaries
        if summary.get("reliability_drop_wrong") is not None
    ]
    misleading_reliability_drops = [
        float(summary["reliability_drop_wrong"])
        for summary, _ in misleading_logs
        if summary.get("reliability_drop_wrong") is not None
    ]
    final_reliabilities = [
        float(summary["final_reliability_wrong"])
        for summary in summaries
        if summary.get("final_reliability_wrong") is not None
    ]
    final_true_support_reliabilities = [
        reliability
        for summary, step_logs in method_logs
        if (
            reliability := get_final_reliability(
                summary,
                step_logs,
                summary.get("true_support_instance_id"),
            )
        )
        is not None
    ]
    true_support_reliability_drops = [
        1.0 - reliability for reliability in final_true_support_reliabilities
    ]

    return {
        "method": method,
        "total_episodes": len(summaries),
        "success_rate": success_rate(summaries),
        "normal_success_rate": success_rate(normal_summaries),
        "misleading_success_rate": success_rate(misleading_summaries),
        "avg_steps": mean(
            [float(summary.get("num_steps", 0)) for summary in summaries]
        ),
        "avg_steps_misleading": mean(
            [
                float(summary.get("num_steps", 0))
                for summary in misleading_summaries
            ]
        ),
        "avg_wrong_inspections_all": mean(
            [
                float(count_wrong_inspections(summary, step_logs))
                for summary, step_logs in method_logs
            ]
        ),
        "avg_wrong_inspections_misleading": mean(
            [
                float(count_wrong_inspections(summary, step_logs))
                for summary, step_logs in misleading_logs
            ]
        ),
        "switch_rate_misleading": mean(
            [
                1.0 if switched_after_wrong(summary, step_logs) else 0.0
                for summary, step_logs in misleading_logs
            ]
        ),
        "avg_reliability_drop_wrong": mean(reliability_drops),
        "avg_reliability_drop_wrong_misleading": mean(
            misleading_reliability_drops
        ),
        "avg_final_reliability_wrong": mean(final_reliabilities),
        "avg_final_reliability_true_support": mean(
            final_true_support_reliabilities
        ),
        "avg_reliability_drop_true_support": mean(
            true_support_reliability_drops
        ),
        "true_support_suppressed_rate": mean(
            [
                1.0 if reliability < 0.5 else 0.0
                for reliability in final_true_support_reliabilities
            ]
        ),
        "avg_spf_trigger_count": mean(
            [float(count_spf_triggers(step_logs)) for _, step_logs in method_logs]
        ),
        "avg_spf_trigger_count_misleading": mean(
            [
                float(count_spf_triggers(step_logs))
                for _, step_logs in misleading_logs
            ]
        ),
    }


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_markdown_table(rows: list[dict[str, Any]]) -> None:
    columns = [
        "method",
        "total_episodes",
        "success_rate",
        "normal_success_rate",
        "misleading_success_rate",
        "avg_steps",
        "avg_steps_misleading",
        "avg_wrong_inspections_all",
        "avg_wrong_inspections_misleading",
        "switch_rate_misleading",
        "avg_reliability_drop_wrong",
        "avg_reliability_drop_wrong_misleading",
        "avg_final_reliability_wrong",
        "avg_final_reliability_true_support",
        "avg_reliability_drop_true_support",
        "true_support_suppressed_rate",
        "avg_spf_trigger_count",
        "avg_spf_trigger_count_misleading",
    ]
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        print(
            "| "
            + " | ".join(format_cell(row.get(column, "")) for column in columns)
            + " |"
        )


def save_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("outputs/logs/debug_batch"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/tables/debug_controlled_results.csv"),
    )
    args = parser.parse_args()

    rows = []
    for method_dir in sorted(path for path in args.log_root.iterdir() if path.is_dir()):
        method_logs = load_method_logs(method_dir)
        if method_logs:
            rows.append(evaluate_method(method_dir.name, method_logs))

    print_markdown_table(rows)
    save_csv(rows, args.output)
    print(f"saved csv: {args.output}")


if __name__ == "__main__":
    main()

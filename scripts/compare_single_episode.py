"""Compare several single-episode controlled-run logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def none_if_missing(value: Any) -> Any:
    return None if value is None else value


def summarize_log_dir(log_dir: Path) -> dict[str, Any]:
    summary = read_json(log_dir / "episode_summary.json")
    step_logs = read_jsonl(log_dir / "step_log.jsonl")
    wrong_instance_id = summary.get("wrong_instance_id")

    wrong_inspections = sum(
        1
        for row in step_logs
        if row.get("selected_instance_id") == wrong_instance_id
        and float(row.get("evidence", 0.0)) == 0.0
    )

    switch_step = None
    for row in step_logs:
        if row.get("selected_instance_id") != wrong_instance_id:
            switch_step = row.get("step")
            break

    spf_trigger_count = sum(
        1 for row in step_logs if bool(row.get("spf_triggered", False))
    )

    selected_alias_sequence = " -> ".join(
        str(row.get("selected_instance_alias", ""))
        for row in step_logs
    )

    return {
        "method": summary.get("method", ""),
        "success": summary.get("success", False),
        "num_steps": summary.get("num_steps", len(step_logs)),
        "selected_alias_sequence": selected_alias_sequence,
        "wrong_inspections": wrong_inspections,
        "switch_step": none_if_missing(switch_step),
        "reliability_drop_wrong": none_if_missing(
            summary.get("reliability_drop_wrong")
        ),
        "final_reliability_wrong": none_if_missing(
            summary.get("final_reliability_wrong")
        ),
        "spf_trigger_count": spf_trigger_count,
    }


def format_cell(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_markdown_table(rows: list[dict[str, Any]]) -> None:
    columns = [
        "method",
        "success",
        "num_steps",
        "selected_alias_sequence",
        "wrong_inspections",
        "switch_step",
        "reliability_drop_wrong",
        "final_reliability_wrong",
        "spf_trigger_count",
    ]
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        print(
            "| "
            + " | ".join(format_cell(row.get(column)) for column in columns)
            + " |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dirs",
        type=Path,
        nargs="+",
        default=[
            Path("outputs/logs/debug_ours_ep1"),
            Path("outputs/logs/debug_sp_greedy_ep1"),
            Path("outputs/logs/debug_visit_ep1"),
        ],
    )
    args = parser.parse_args()

    rows = [summarize_log_dir(log_dir) for log_dir in args.log_dirs]
    print_markdown_table(rows)


if __name__ == "__main__":
    main()

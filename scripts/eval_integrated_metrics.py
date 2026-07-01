"""Evaluate integrated ObjectNav batch logs by method."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean
from typing import Any


DEFAULT_TAU_E = 0.1
NAVIGATION_FORCED_SWITCH_REASONS = {"navigation_failed", "accessibility_zero"}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def load_method_logs(
    method_dir: Path,
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    logs = []
    for summary_path in sorted(method_dir.glob("*/episode_summary.json")):
        step_log_path = summary_path.parent / "step_log.jsonl"
        step_logs = read_jsonl(step_log_path) if step_log_path.exists() else []
        logs.append((read_json(summary_path), step_logs))
    return logs


def load_episode_metadata(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    metadata = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            episode = json.loads(line)
            episode_id = episode.get("episode_id")
            if episode_id is None:
                continue
            metadata[str(episode_id)] = {
                "episode_type": episode.get("episode_type"),
                "target_category": episode.get("target_category"),
                "scene": episode.get("scene"),
                "wrong_instance_id": episode.get("wrong_instance_id"),
                "true_support_instance_id": episode.get(
                    "true_support_instance_id"
                ),
            }
    return metadata


def apply_episode_metadata(
    method_logs: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    episode_metadata: dict[str, dict[str, Any]],
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    if not episode_metadata:
        return method_logs

    enriched_logs = []
    fields = [
        "episode_type",
        "target_category",
        "scene",
        "wrong_instance_id",
        "true_support_instance_id",
    ]
    for summary, step_logs in method_logs:
        summary = dict(summary)
        episode_id = summary.get("episode_id")
        metadata = episode_metadata.get(str(episode_id), {})
        for field in fields:
            if summary.get(field) is None and metadata.get(field) is not None:
                summary[field] = metadata[field]
        enriched_logs.append((summary, step_logs))
    return enriched_logs


def is_system_error(summary: dict[str, Any]) -> bool:
    if "error" in summary:
        return True
    success = bool(summary.get("success", False))
    num_actions = int(summary.get("num_actions", summary.get("num_steps", 0)) or 0)
    num_decisions = int(summary.get("num_decisions", 0) or 0)
    return num_actions == 0 and num_decisions == 0 and not success


def success_rate(summaries: list[dict[str, Any]]) -> float:
    return mean([1.0 if summary.get("success", False) else 0.0 for summary in summaries])


def success_is_true_support(summary: dict[str, Any]) -> bool:
    if summary.get("success_is_true_support") is not None:
        return safe_bool(summary.get("success_is_true_support"))
    success_selected_id = summary.get("success_selected_instance_id")
    true_support_id = summary.get("true_support_instance_id")
    return (
        bool(summary.get("success", False))
        and success_selected_id is not None
        and success_selected_id == true_support_id
    )


def visible_before_true_support(summary: dict[str, Any]) -> bool:
    if summary.get("visible_before_true_support") is not None:
        return safe_bool(summary.get("visible_before_true_support"))
    return (
        bool(summary.get("success", False))
        and safe_bool(summary.get("target_seen_any", False))
        and not safe_bool(summary.get("true_support_selected", False))
    )


def non_true_support_visible_success(summary: dict[str, Any]) -> bool:
    return bool(summary.get("success", False)) and not success_is_true_support(summary)


def decision_rows(step_logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in step_logs if row.get("phase") == "decision_update"]
    if rows:
        return rows
    rows = [row for row in step_logs if row.get("action") == "Inspect"]
    if rows:
        return rows

    by_decision: dict[Any, dict[str, Any]] = {}
    for row in step_logs:
        decision_step = row.get("decision_step")
        if decision_step is not None:
            by_decision[decision_step] = row
    return [by_decision[key] for key in sorted(by_decision)]


def fallback_num_actions(step_logs: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in step_logs
        if row.get("phase") != "decision_update"
        and row.get("phase") != "virtual_stop"
        and row.get("action_type") != "virtual"
        and row.get("simulator_executed", True) is not False
    )


def get_num_actions(summary: dict[str, Any], step_logs: list[dict[str, Any]]) -> float:
    if summary.get("num_actions") is not None:
        return safe_float(summary.get("num_actions"))
    return float(fallback_num_actions(step_logs))


def get_num_decisions(summary: dict[str, Any], step_logs: list[dict[str, Any]]) -> float:
    if summary.get("num_decisions") is not None:
        return safe_float(summary.get("num_decisions"))
    return float(len(decision_rows(step_logs)))


def get_path_length(summary: dict[str, Any], step_logs: list[dict[str, Any]]) -> float:
    if summary.get("path_length") is not None:
        return safe_float(summary.get("path_length"))
    path_lengths = [
        safe_float(row.get("path_length"))
        for row in step_logs
        if row.get("path_length") is not None
    ]
    return max(path_lengths) if path_lengths else 0.0


def count_wrong_prior_decisions(
    summary: dict[str, Any],
    step_logs: list[dict[str, Any]],
) -> int:
    if summary.get("wrong_prior_decisions") is not None:
        return int(summary.get("wrong_prior_decisions") or 0)

    wrong_instance_id = summary.get("wrong_instance_id")
    if wrong_instance_id is None:
        return 0
    return sum(
        1
        for row in decision_rows(step_logs)
        if row.get("selected_instance_id") == wrong_instance_id
    )


def selected_sequence_from_logs(step_logs: list[dict[str, Any]]) -> list[str]:
    sequence = []
    for row in decision_rows(step_logs):
        selected_id = row.get("selected_instance_id")
        if selected_id is not None:
            sequence.append(str(selected_id))
    return sequence


def selected_sequence(
    summary: dict[str, Any], step_logs: list[dict[str, Any]]
) -> list[str]:
    sequence = summary.get("selected_sequence")
    if isinstance(sequence, list):
        return [str(instance_id) for instance_id in sequence]
    return selected_sequence_from_logs(step_logs)


def switch_diagnostics_from_sequence(
    sequence: list[str], wrong_instance_id: str | None
) -> tuple[bool, float | None]:
    if wrong_instance_id is None:
        return False, None

    first_wrong_index = None
    for index, instance_id in enumerate(sequence):
        if instance_id == wrong_instance_id and first_wrong_index is None:
            first_wrong_index = index
        elif first_wrong_index is not None and instance_id != wrong_instance_id:
            return True, float(index - first_wrong_index)
    return False, None


def switch_events(
    summary: dict[str, Any], step_logs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    summary_events = summary.get("switch_events")
    if isinstance(summary_events, list):
        return [
            event
            for event in summary_events
            if isinstance(event, dict)
            and event.get("switch_from") is not None
            and event.get("switch_to") is not None
        ]

    events = []
    previous_instance_id = None
    for row in decision_rows(step_logs):
        selected_instance_id = row.get("selected_instance_id")
        if selected_instance_id is None:
            continue
        selected_instance_id = str(selected_instance_id)
        if (
            previous_instance_id is not None
            and selected_instance_id != previous_instance_id
        ):
            reason = row.get("switch_reason") or "utility_change"
            navigation_forced = safe_bool(
                row.get(
                    "navigation_forced_switch",
                    reason in NAVIGATION_FORCED_SWITCH_REASONS,
                )
            )
            events.append(
                {
                    "decision_step": row.get("decision_step"),
                    "switch_from": previous_instance_id,
                    "switch_to": selected_instance_id,
                    "switch_reason": reason,
                    "navigation_forced_switch": navigation_forced,
                }
            )
        previous_instance_id = selected_instance_id
    return events


def first_switch_from_wrong_event(
    summary: dict[str, Any], step_logs: list[dict[str, Any]]
) -> dict[str, Any] | None:
    wrong_instance_id = summary.get("wrong_instance_id")
    if wrong_instance_id is None:
        return None
    wrong_instance_id = str(wrong_instance_id)
    for event in switch_events(summary, step_logs):
        if (
            str(event.get("switch_from")) == wrong_instance_id
            and str(event.get("switch_to")) != wrong_instance_id
        ):
            return event
    return None


def switched_after_wrong_policy(
    summary: dict[str, Any],
    step_logs: list[dict[str, Any]],
) -> bool:
    if summary.get("switched_from_wrong_policy") is not None:
        return safe_bool(summary.get("switched_from_wrong_policy"))
    event = first_switch_from_wrong_event(summary, step_logs)
    return bool(event) and not safe_bool(event.get("navigation_forced_switch"))


def navigation_forced_switch_after_wrong(
    summary: dict[str, Any],
    step_logs: list[dict[str, Any]],
) -> bool:
    if summary.get("navigation_forced_switch") is not None:
        return safe_bool(summary.get("navigation_forced_switch"))
    event = first_switch_from_wrong_event(summary, step_logs)
    return bool(event) and safe_bool(event.get("navigation_forced_switch"))


def switched_after_wrong(
    summary: dict[str, Any],
    step_logs: list[dict[str, Any]],
) -> bool:
    wrong_instance_id = summary.get("wrong_instance_id")
    fallback_switched, _ = switch_diagnostics_from_sequence(
        selected_sequence(summary, step_logs),
        None if wrong_instance_id is None else str(wrong_instance_id),
    )
    if summary.get("switched_from_wrong") is None:
        return fallback_switched
    return safe_bool(summary.get("switched_from_wrong"))


def time_to_switch(
    summary: dict[str, Any],
    step_logs: list[dict[str, Any]],
) -> float | None:
    wrong_instance_id = summary.get("wrong_instance_id")
    _, fallback_switch_time = switch_diagnostics_from_sequence(
        selected_sequence(summary, step_logs),
        None if wrong_instance_id is None else str(wrong_instance_id),
    )
    if summary.get("time_to_switch") is None:
        return fallback_switch_time
    return safe_float(summary.get("time_to_switch"))


def count_spf_triggers(step_logs: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in decision_rows(step_logs)
        if bool(row.get("spf_triggered", False))
    )


def get_spf_trigger_count(
    summary: dict[str, Any], step_logs: list[dict[str, Any]]
) -> int:
    if summary.get("spf_trigger_count") is not None:
        return int(summary.get("spf_trigger_count") or 0)
    return count_spf_triggers(step_logs)


def get_reliability_drop_wrong(
    summary: dict[str, Any], step_logs: list[dict[str, Any]]
) -> float | None:
    if summary.get("reliability_drop_wrong") is not None:
        return safe_float(summary.get("reliability_drop_wrong"))

    wrong_instance_id = summary.get("wrong_instance_id")
    if wrong_instance_id is None:
        return None

    final_reliability = summary.get("final_reliability_wrong")
    if final_reliability is None:
        for row in reversed(step_logs):
            if row.get("selected_instance_id") == wrong_instance_id:
                final_reliability = row.get("reliability_after", row.get("reliability"))
                break
    if final_reliability is None:
        return None
    return 1.0 - safe_float(final_reliability, 1.0)


def get_stop_precision(summary: dict[str, Any], step_logs: list[dict[str, Any]]) -> float:
    if summary.get("stop_precision") is not None:
        return safe_float(summary.get("stop_precision"))

    stopped = bool(summary.get("stopped", False))
    if not stopped:
        stopped = any(row.get("phase") == "virtual_stop" for row in step_logs)
    success = bool(summary.get("success", False))
    return 1.0 if stopped and success else 0.0


def evaluate_method(
    method: str,
    method_logs: list[tuple[dict[str, Any], list[dict[str, Any]]]],
) -> dict[str, Any]:
    summaries = [summary for summary, _ in method_logs]
    valid_logs = [
        (summary, step_logs)
        for summary, step_logs in method_logs
        if not is_system_error(summary)
    ]
    valid_summaries = [summary for summary, _ in valid_logs]
    normal_summaries = [
        summary
        for summary in valid_summaries
        if summary.get("episode_type") == "normal-prior"
    ]
    misleading_summaries = [
        summary
        for summary in valid_summaries
        if summary.get("episode_type") == "misleading-prior"
    ]
    misleading_logs = [
        (summary, step_logs)
        for summary, step_logs in valid_logs
        if summary.get("episode_type") == "misleading-prior"
    ]

    reliability_drops = [
        reliability_drop
        for summary, step_logs in valid_logs
        if (
            reliability_drop := get_reliability_drop_wrong(summary, step_logs)
        )
        is not None
    ]
    switch_times = [
        switch_time
        for summary, step_logs in misleading_logs
        if switched_after_wrong(summary, step_logs)
        and (switch_time := time_to_switch(summary, step_logs)) is not None
    ]
    switch_rate_misleading_all = mean(
        [
            1.0 if switched_after_wrong(summary, step_logs) else 0.0
            for summary, step_logs in misleading_logs
        ]
    )
    switch_rate_misleading_policy = mean(
        [
            1.0 if switched_after_wrong_policy(summary, step_logs) else 0.0
            for summary, step_logs in misleading_logs
        ]
    )
    navigation_forced_switch_rate = mean(
        [
            1.0 if navigation_forced_switch_after_wrong(summary, step_logs) else 0.0
            for summary, step_logs in misleading_logs
        ]
    )

    return {
        "method": method,
        "total_episodes": len(summaries),
        "error_count": sum(1 for summary in summaries if is_system_error(summary)),
        "valid_episodes": len(valid_summaries),
        "success_rate_valid": success_rate(valid_summaries),
        "true_support_success_rate": mean(
            [
                1.0 if success_is_true_support(summary) else 0.0
                for summary in valid_summaries
            ]
        ),
        "visible_before_true_support_rate": mean(
            [
                1.0 if visible_before_true_support(summary) else 0.0
                for summary in valid_summaries
            ]
        ),
        "non_true_support_visible_success_rate": mean(
            [
                1.0 if non_true_support_visible_success(summary) else 0.0
                for summary in valid_summaries
            ]
        ),
        "normal_success_rate_valid": success_rate(normal_summaries),
        "misleading_success_rate_valid": success_rate(misleading_summaries),
        "avg_actions": mean(
            [get_num_actions(summary, step_logs) for summary, step_logs in valid_logs]
        ),
        "avg_decisions": mean(
            [get_num_decisions(summary, step_logs) for summary, step_logs in valid_logs]
        ),
        "avg_path_length": mean(
            [get_path_length(summary, step_logs) for summary, step_logs in valid_logs]
        ),
        "avg_wrong_prior_decisions": mean(
            [
                float(count_wrong_prior_decisions(summary, step_logs))
                for summary, step_logs in valid_logs
            ]
        ),
        "switch_rate_misleading": switch_rate_misleading_all,
        "switch_rate_misleading_all": switch_rate_misleading_all,
        "switch_rate_misleading_policy": switch_rate_misleading_policy,
        "navigation_forced_switch_rate": navigation_forced_switch_rate,
        "avg_time_to_switch": mean(switch_times),
        "avg_reliability_drop_wrong": mean(reliability_drops),
        "avg_spf_trigger_count": mean(
            [
                float(get_spf_trigger_count(summary, step_logs))
                for summary, step_logs in valid_logs
            ]
        ),
        "stop_precision": mean(
            [get_stop_precision(summary, step_logs) for summary, step_logs in valid_logs]
        ),
    }


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def output_columns() -> list[str]:
    return [
        "method",
        "total_episodes",
        "error_count",
        "valid_episodes",
        "success_rate_valid",
        "true_support_success_rate",
        "visible_before_true_support_rate",
        "non_true_support_visible_success_rate",
        "normal_success_rate_valid",
        "misleading_success_rate_valid",
        "avg_actions",
        "avg_decisions",
        "avg_path_length",
        "avg_wrong_prior_decisions",
        "switch_rate_misleading",
        "switch_rate_misleading_all",
        "switch_rate_misleading_policy",
        "navigation_forced_switch_rate",
        "avg_time_to_switch",
        "avg_reliability_drop_wrong",
        "avg_spf_trigger_count",
        "stop_precision",
    ]


def print_markdown_table(rows: list[dict[str, Any]]) -> None:
    columns = output_columns()
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
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=output_columns())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("outputs/logs/integrated_batch"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/tables/integrated_results.csv"),
    )
    parser.add_argument("--episodes", type=str, default=None)
    args = parser.parse_args()

    episode_metadata = load_episode_metadata(
        Path(args.episodes) if args.episodes is not None else None
    )
    rows = []
    if args.log_root.exists():
        for method_dir in sorted(
            path for path in args.log_root.iterdir() if path.is_dir()
        ):
            method_logs = load_method_logs(method_dir)
            method_logs = apply_episode_metadata(method_logs, episode_metadata)
            if method_logs:
                rows.append(evaluate_method(method_dir.name, method_logs))

    print_markdown_table(rows)
    save_csv(rows, args.output)
    print(f"saved csv: {args.output}")


if __name__ == "__main__":
    main()

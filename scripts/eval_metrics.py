"""Evaluate navigation and diagnostic metrics from saved episode logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from statistics import fmean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from iac_zson.metrics.diagnostic_metrics import (
    compute_reliability_drop,
    compute_rsr,
    compute_switch_rate,
    compute_tts,
    compute_wpfs,
    compute_wrong_inspections,
)
from iac_zson.metrics.nav_metrics import compute_spl, compute_sr


def _mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def load_logs(log_root: Path) -> tuple[list[dict], list[list[dict]]]:
    """Load summaries and their sibling step logs recursively."""
    summaries = []
    episode_step_logs = []
    for summary_path in sorted(log_root.rglob("episode_summary.json")):
        step_path = summary_path.parent / "step_log.jsonl"
        if not step_path.exists():
            continue

        with summary_path.open("r", encoding="utf-8") as file:
            summaries.append(json.load(file))
        with step_path.open("r", encoding="utf-8") as file:
            episode_step_logs.append(
                [json.loads(line) for line in file if line.strip()]
            )
    return summaries, episode_step_logs


def evaluate(
    log_root: Path, tau_c: float = 0.6, tau_e: float = 0.1
) -> dict[str, float]:
    summaries, episode_step_logs = load_logs(log_root)

    spl_values = [
        compute_spl(
            bool(summary.get("success", False)),
            float(summary.get("path_length", 0.0) or 0.0),
            float(summary.get("shortest_path_length", 0.0) or 0.0),
        )
        for summary in summaries
    ]
    tts_values = [
        tts
        for logs in episode_step_logs
        if (tts := compute_tts(logs)) is not None
    ]

    return {
        "SR": compute_sr(summaries),
        "SPL": _mean(spl_values),
        "WPFS": _mean(
            [compute_wpfs(logs, tau_c, tau_e) for logs in episode_step_logs]
        ),
        "RSR": _mean([compute_rsr(logs) for logs in episode_step_logs]),
        "TTS": _mean(tts_values),
        "Wrong Inspections": _mean(
            [
                float(compute_wrong_inspections(logs, tau_c, tau_e))
                for logs in episode_step_logs
            ]
        ),
        "Switch Rate": compute_switch_rate(summaries),
        "Reliability Drop": _mean(
            [compute_reliability_drop(logs) for logs in episode_step_logs]
        ),
    }


def save_results(results: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results))
        writer.writeheader()
        writer.writerow(results)


def main() -> dict[str, float]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "log_root",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "logs",
    )
    parser.add_argument("--tau-c", type=float, default=0.6)
    parser.add_argument("--tau-e", type=float, default=0.1)
    args = parser.parse_args()

    results = evaluate(args.log_root, args.tau_c, args.tau_e)
    output_path = PROJECT_ROOT / "outputs" / "tables" / "mock_results.csv"
    save_results(results, output_path)
    print(results)
    print(output_path)
    return results


if __name__ == "__main__":
    main()


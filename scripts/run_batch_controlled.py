"""Run controlled-inspection single-episode logic over a small episode set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai2thor.controller import Controller

from iac_zson.logging.episode_logger import EpisodeLogger
from iac_zson.planning.controlled_inspector import ControlledInspector
from scripts.run_controlled_episode import (
    METHOD_LABELS,
    METHOD_CHOICES,
    initialize_memory,
    run_decisions,
)


def load_episodes(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def summary_path(output_root: Path, method: str, episode_id: str) -> Path:
    return output_root / method / episode_id / "episode_summary.json"


def write_error_summary(
    output_root: Path,
    method: str,
    episode: dict[str, Any],
    error: Exception,
) -> dict[str, Any]:
    summary = {
        "episode_id": episode["episode_id"],
        "method": METHOD_LABELS[method],
        "success": False,
        "error": str(error),
        "num_steps": 0,
        "episode_type": episode.get("episode_type"),
        "target_category": episode.get("target_category"),
        "wrong_instance_id": episode.get("wrong_instance_id"),
        "true_support_instance_id": episode.get("true_support_instance_id"),
    }
    output_path = summary_path(output_root, method, episode["episode_id"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return summary


def run_one_episode(
    episode: dict[str, Any],
    method: str,
    output_root: Path,
    max_decisions: int,
    tau_c: float,
    tau_n: int,
    beta_v: float,
    quiet: bool,
    partial_inspection: bool,
    poses_per_decision: int,
) -> dict[str, Any]:
    memory = initialize_memory(
        episode,
        tau_c=tau_c,
        tau_e=0.1,
        tau_n=tau_n,
        lambda_inst=0.35,
        alpha=0.25,
    )
    controller = None
    logger = EpisodeLogger(
        output_root / method,
        episode["episode_id"],
        METHOD_LABELS[method],
    )
    try:
        controller = Controller(
            scene=episode["scene"],
            width=300,
            height=300,
            gridSize=0.25,
            renderDepthImage=False,
            renderInstanceSegmentation=False,
        )
        inspector = ControlledInspector(controller)
        summary = run_decisions(
            episode=episode,
            memory=memory,
            inspector=inspector,
            logger=logger,
            max_decisions=max_decisions,
            method=method,
            beta_v=beta_v,
            quiet=quiet,
            partial_inspection=partial_inspection,
            poses_per_decision=poses_per_decision,
        )
        logger.save_summary(summary)
        return summary
    finally:
        if controller is not None:
            controller.stop()
        logger.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes",
        type=Path,
        default=PROJECT_ROOT / "data" / "episodes" / "debug_episodes.jsonl",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHOD_CHOICES,
        default=["ours", "sp_greedy", "sp_visit_penalty"],
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "logs" / "debug_batch",
    )
    parser.add_argument("--max-decisions", type=int, default=10)
    parser.add_argument("--tau-c", type=float, default=0.6)
    parser.add_argument("--tau-n", type=int, default=1)
    parser.add_argument("--beta-v", type=float, default=0.7)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--partial-inspection", action="store_true")
    parser.add_argument("--poses-per-decision", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    episodes = load_episodes(args.episodes)
    success_counts = {method: 0 for method in args.methods}

    for method in args.methods:
        if not args.quiet:
            print(f"running method: {method}")
        for episode in episodes:
            episode_id = episode["episode_id"]
            if args.resume and summary_path(
                args.output_root, method, episode_id
            ).exists():
                print(f"[{method}] skipped {episode_id} resume=True")
                continue

            print(f"[{method}] starting {episode_id}")
            try:
                summary = run_one_episode(
                    episode=episode,
                    method=method,
                    output_root=args.output_root,
                    max_decisions=args.max_decisions,
                    tau_c=args.tau_c,
                    tau_n=args.tau_n,
                    beta_v=args.beta_v,
                    quiet=args.quiet,
                    partial_inspection=args.partial_inspection,
                    poses_per_decision=args.poses_per_decision,
                )
                print(
                    f"[{method}] finished {episode_id} "
                    f"success={summary['success']} "
                    f"num_steps={summary['num_steps']}"
                )
            except Exception as error:
                summary = write_error_summary(
                    args.output_root, method, episode, error
                )
                print(
                    f"[{method}] failed {episode_id} "
                    f"success=False num_steps=0 error={error}"
                )
            if bool(summary["success"]):
                success_counts[method] += 1

    print("success counts:")
    for method in args.methods:
        print(f"{method}: {success_counts[method]}/{len(episodes)}")


if __name__ == "__main__":
    main()

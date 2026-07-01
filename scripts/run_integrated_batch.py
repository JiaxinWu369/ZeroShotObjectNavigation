"""Batch wrapper for the integrated ObjectNav episode runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from iac_zson.logging.episode_logger import EpisodeLogger
from scripts.run_integrated_episode import METHOD_LABELS, run_episode, teleport_to_start


def load_episodes(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def make_error_summary(
    episode: dict[str, Any],
    method: str,
    error: Exception,
) -> dict[str, Any]:
    return {
        "episode_id": episode.get("episode_id"),
        "method": METHOD_LABELS.get(method, method),
        "success": False,
        "error": str(error),
        "num_actions": 0,
        "num_decisions": 0,
        "episode_type": episode.get("episode_type"),
        "target_category": episode.get("target_category"),
        "wrong_instance_id": episode.get("wrong_instance_id"),
        "true_support_instance_id": episode.get("true_support_instance_id"),
    }


def run_one(
    episode: dict[str, Any],
    method: str,
    output_root: Path,
    max_steps: int,
    max_decisions: int,
    tau_c: float,
    tau_n: int,
    beta_v: float,
    partial_inspection: bool,
    poses_per_decision: int,
) -> dict[str, Any]:
    episode_id = episode["episode_id"]
    logger = EpisodeLogger(output_root / method, episode_id, METHOD_LABELS[method])
    controller = None
    try:
        from ai2thor.controller import Controller

        controller = Controller(
            scene=episode["scene"],
            width=300,
            height=300,
            gridSize=0.25,
            renderDepthImage=False,
            renderInstanceSegmentation=False,
        )
        teleport_to_start(controller, episode)
        summary = run_episode(
            episode=episode,
            controller=controller,
            logger=logger,
            method=method,
            max_steps=max_steps,
            max_decisions=max_decisions,
            tau_c=tau_c,
            tau_e=0.1,
            tau_n=tau_n,
            partial_inspection=partial_inspection,
            poses_per_decision=poses_per_decision,
            beta_v=beta_v,
        )
        logger.save_summary(summary)
        return summary
    except Exception as error:
        summary = make_error_summary(episode, method, error)
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
        choices=("ours", "sp_greedy", "sp_visit_penalty"),
        default=["ours", "sp_greedy", "sp_visit_penalty"],
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "logs" / "integrated_batch",
    )
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--max-decisions", type=int, default=30)
    parser.add_argument("--tau-c", type=float, default=0.75)
    parser.add_argument("--tau-n", type=int, default=2)
    parser.add_argument("--beta-v", type=float, default=0.7)
    parser.add_argument("--partial-inspection", action="store_true")
    parser.add_argument("--poses-per-decision", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    episodes = load_episodes(args.episodes)
    success_counts = {method: 0 for method in args.methods}

    for method in args.methods:
        for episode in episodes:
            episode_id = episode["episode_id"]
            episode_dir = args.output_root / method / episode_id
            summary_path = episode_dir / "episode_summary.json"

            if args.resume and summary_path.exists():
                if not args.quiet:
                    print(f"[{method}] skipped {episode_id} resume=True")
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    if summary.get("success", False):
                        success_counts[method] += 1
                except json.JSONDecodeError:
                    pass
                continue

            if not args.quiet:
                print(f"[{method}] starting {episode_id}")

            summary = run_one(
                episode=episode,
                method=method,
                output_root=args.output_root,
                max_steps=args.max_steps,
                max_decisions=args.max_decisions,
                tau_c=args.tau_c,
                tau_n=args.tau_n,
                beta_v=args.beta_v,
                partial_inspection=args.partial_inspection,
                poses_per_decision=args.poses_per_decision,
            )

            if summary.get("success", False):
                success_counts[method] += 1

            if not args.quiet:
                print(
                    f"[{method}] finished {episode_id} "
                    f"success={summary.get('success', False)} "
                    f"num_actions={summary.get('num_actions', 0)} "
                    f"num_decisions={summary.get('num_decisions', 0)}"
                )

    for method in args.methods:
        print(f"[{method}] success count: {success_counts[method]}/{len(episodes)}")


if __name__ == "__main__":
    main()

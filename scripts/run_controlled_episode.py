"""Run one Ours episode with teleport-based controlled inspection."""

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
from iac_zson.memory.instance_memory import InstanceState, SemanticInstanceMemory
from iac_zson.planning.controlled_inspector import ControlledInspector


def load_episode(path: Path, episode_index: int) -> dict[str, Any]:
    if episode_index < 0:
        raise IndexError("episode index must be non-negative")
    with path.open("r", encoding="utf-8") as file:
        episodes = [json.loads(line) for line in file if line.strip()]
    return episodes[episode_index]


def initialize_memory(
    episode: dict[str, Any],
    tau_c: float,
    tau_e: float,
    tau_n: int,
    lambda_inst: float,
    alpha: float,
) -> SemanticInstanceMemory:
    memory = SemanticInstanceMemory(
        tau_c=tau_c,
        tau_e=tau_e,
        tau_n=tau_n,
        lambda_inst=lambda_inst,
        alpha=alpha,
    )
    for candidate in episode["candidate_instances"]:
        position = candidate["position"]
        memory.add_instance(
            InstanceState(
                instance_id=candidate["instance_id"],
                alias=candidate["alias"],
                category=candidate["category"],
                region_center=(
                    float(position["x"]),
                    float(position.get("y", 0.0)),
                    float(position["z"]),
                ),
                p_sem=float(candidate["p_sem"]),
                reliability=1.0,
                coverage=0.0,
                evidence=0.0,
                inspect_count=0,
            )
        )
    return memory


def run_decisions(
    episode: dict[str, Any],
    memory: SemanticInstanceMemory,
    inspector: ControlledInspector,
    logger: Any,
    max_decisions: int,
) -> dict[str, Any]:
    candidates_by_id = {
        candidate["instance_id"]: candidate
        for candidate in episode["candidate_instances"]
    }
    selected_sequence = []
    success = False

    for step in range(max_decisions):
        accessibility = {
            instance.instance_id: 1.0 for instance in memory.all_instances()
        }
        information_gain = {
            instance.instance_id: max(0.5, 1.0 - instance.coverage)
            for instance in memory.all_instances()
        }
        selected = memory.select_best_instance(
            accessibility, information_gain, step=step
        )
        if selected is None:
            break

        utility = memory.compute_utility(
            selected.instance_id,
            accessibility=1.0,
            information_gain=information_gain[selected.instance_id],
        )
        reliability_before = selected.reliability
        observation_info = inspector.inspect(
            episode, candidates_by_id[selected.instance_id]
        )

        memory.update_coverage(
            selected.instance_id,
            max(selected.coverage, float(observation_info["coverage"])),
        )
        memory.update_evidence(
            selected.instance_id,
            max(selected.evidence, float(observation_info["evidence"])),
        )
        memory.finish_inspection(
            selected.instance_id,
            observation_info.get("visited_viewpoint_id"),
        )
        spf_triggered = memory.detect_spf(selected.instance_id)
        reliability_after = memory.update_reliability(selected.instance_id)

        success = (
            selected.instance_id == episode["true_support_instance_id"]
            and float(observation_info["evidence"]) == 1.0
        )
        selected_sequence.append(selected.instance_id)
        row = {
            "episode_id": episode["episode_id"],
            "method": "Ours",
            "step": step,
            "target_category": episode["target_category"],
            "selected_instance_id": selected.instance_id,
            "selected_instance_alias": selected.alias,
            "selected_instance_category": selected.category,
            "p_sem": selected.p_sem,
            "accessibility": 1.0,
            "information_gain": information_gain[selected.instance_id],
            "utility": utility,
            "coverage": selected.coverage,
            "evidence": selected.evidence,
            "inspect_count": selected.inspect_count,
            "reliability_before": reliability_before,
            "reliability_after": reliability_after,
            "spf_triggered": spf_triggered,
            "target_visible": bool(observation_info["target_visible"]),
            "success": success,
            "finish_inspection": True,
            "visited_viewpoint_id": observation_info.get(
                "visited_viewpoint_id", ""
            ),
            "num_inspection_poses": observation_info[
                "num_inspection_poses"
            ],
        }
        logger.log_step(row)
        print(
            f"step={step} selected_alias={selected.alias} "
            f"selected_category={selected.category} p_sem={selected.p_sem:.4f} "
            f"coverage={selected.coverage:.4f} evidence={selected.evidence:.4f} "
            f"inspect_count={selected.inspect_count} "
            f"reliability_before={reliability_before:.4f} "
            f"reliability_after={reliability_after:.4f} utility={utility:.4f} "
            f"spf_triggered={spf_triggered} success={success}"
        )

        if success:
            break

    wrong_instance_id = episode.get("wrong_instance_id")
    if wrong_instance_id is not None and wrong_instance_id in candidates_by_id:
        final_reliability_wrong = memory.get_instance(
            wrong_instance_id
        ).reliability
        reliability_drop_wrong = 1.0 - final_reliability_wrong
    else:
        final_reliability_wrong = None
        reliability_drop_wrong = None

    return {
        "episode_id": episode["episode_id"],
        "target_category": episode["target_category"],
        "episode_type": episode["episode_type"],
        "success": success,
        "num_steps": len(selected_sequence),
        "selected_sequence": selected_sequence,
        "wrong_instance_id": wrong_instance_id,
        "true_support_instance_id": episode["true_support_instance_id"],
        "final_reliability_wrong": final_reliability_wrong,
        "reliability_drop_wrong": reliability_drop_wrong,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes",
        type=Path,
        default=PROJECT_ROOT / "data" / "episodes" / "debug_episodes.jsonl",
    )
    parser.add_argument("--episode-index", type=int, default=1)
    parser.add_argument("--max-decisions", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "logs" / "debug_ours_ep1",
    )
    parser.add_argument("--tau-c", type=float, default=0.6)
    parser.add_argument("--tau-e", type=float, default=0.1)
    parser.add_argument("--tau-n", type=int, default=1)
    parser.add_argument("--lambda-inst", type=float, default=0.35)
    parser.add_argument("--alpha", type=float, default=0.25)
    args = parser.parse_args()

    episode = load_episode(args.episodes, args.episode_index)
    memory = initialize_memory(
        episode,
        tau_c=args.tau_c,
        tau_e=args.tau_e,
        tau_n=args.tau_n,
        lambda_inst=args.lambda_inst,
        alpha=args.alpha,
    )
    logger = EpisodeLogger(args.output.parent, args.output.name, "Ours")
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
        inspector = ControlledInspector(controller)
        summary = run_decisions(
            episode=episode,
            memory=memory,
            inspector=inspector,
            logger=logger,
            max_decisions=args.max_decisions,
        )
        logger.save_summary(summary)
        print(f"step log: {logger.step_log_path}")
        print(f"episode summary: {logger.summary_path}")
    finally:
        if controller is not None:
            controller.stop()
        logger.close()


if __name__ == "__main__":
    main()

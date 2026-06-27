"""Run one controlled-inspection episode."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from iac_zson.logging.episode_logger import EpisodeLogger
from iac_zson.memory.instance_memory import InstanceState, SemanticInstanceMemory
from iac_zson.planning.controlled_inspector import ControlledInspector


METHOD_LABELS = {
    "ours": "Ours",
    "sp_greedy": "SP-Greedy",
    "sp_visit_penalty": "SP+VisitPenalty",
    "ours_no_coverage": "Ours-NoCoverage",
    "ours_no_repeat": "Ours-NoRepeat",
    "ours_no_category_preserve": "Ours-NoCategoryPreserve",
    "ours_no_reliability": "Ours-NoReliability",
}

OURS_METHODS = {
    "ours",
    "ours_no_coverage",
    "ours_no_repeat",
    "ours_no_category_preserve",
    "ours_no_reliability",
}

METHOD_CHOICES = tuple(METHOD_LABELS)


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
        instance = InstanceState(
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
        instance.visited_viewpoint_ids = set()
        instance.total_inspection_poses = 0
        instance.active_inspection = False
        memory.add_instance(instance)
    return memory


def compute_candidate_scores(
    episode: dict[str, Any],
    memory: SemanticInstanceMemory,
    information_gain: dict[str, float],
    method: str,
    visit_counts: dict[str, int],
    beta_v: float,
) -> list[dict[str, Any]]:
    wrong_instance_id = episode.get("wrong_instance_id")
    true_support_instance_id = episode["true_support_instance_id"]
    scores = []
    for candidate in episode["candidate_instances"]:
        instance = memory.get_instance(candidate["instance_id"])
        visit_count = visit_counts.get(instance.instance_id, 0)
        if method == "sp_greedy":
            utility = instance.p_sem
        elif method == "sp_visit_penalty":
            utility = instance.p_sem * math.exp(-beta_v * visit_count)
        else:
            utility = memory.compute_utility(
                instance.instance_id,
                accessibility=1.0,
                information_gain=information_gain[instance.instance_id],
            )
        scores.append(
            {
                "instance_id": instance.instance_id,
                "alias": instance.alias,
                "category": instance.category,
                "p_sem": instance.p_sem,
                "reliability": instance.reliability,
                "coverage": instance.coverage,
                "evidence": instance.evidence,
                "inspect_count": instance.inspect_count,
                "visit_count": visit_count,
                "information_gain": information_gain[instance.instance_id],
                "utility": utility,
                "is_wrong_instance": instance.instance_id == wrong_instance_id,
                "is_true_support": (
                    instance.instance_id == true_support_instance_id
                ),
            }
        )
    return sorted(scores, key=lambda item: item["utility"], reverse=True)


def select_instance(
    episode: dict[str, Any],
    memory: SemanticInstanceMemory,
    accessibility: dict[str, float],
    information_gain: dict[str, float],
    step: int,
    method: str,
    visit_counts: dict[str, int],
    beta_v: float,
) -> InstanceState | None:
    if method in OURS_METHODS:
        return memory.select_best_instance(
            accessibility, information_gain, step=step
        )

    if method == "sp_greedy":
        best_instance = None
        best_p_sem = float("-inf")
        for candidate in episode["candidate_instances"]:
            instance = memory.get_instance(candidate["instance_id"])
            if instance.p_sem > best_p_sem:
                best_instance = instance
                best_p_sem = instance.p_sem
        return best_instance

    if method == "sp_visit_penalty":
        best_instance = None
        best_utility = float("-inf")
        for candidate in episode["candidate_instances"]:
            instance = memory.get_instance(candidate["instance_id"])
            visit_count = visit_counts.get(instance.instance_id, 0)
            utility = instance.p_sem * math.exp(-beta_v * visit_count)
            if utility > best_utility:
                best_instance = instance
                best_utility = utility
        return best_instance

    raise ValueError(f"unsupported method: {method}")


def detect_spf_for_method(
    instance: InstanceState,
    memory: SemanticInstanceMemory,
    method: str,
) -> bool:
    if method == "ours_no_coverage":
        spf_triggered = (
            instance.evidence <= memory.tau_e
            and instance.inspect_count >= memory.tau_n
        )
    elif method == "ours_no_repeat":
        spf_triggered = (
            instance.coverage >= memory.tau_c
            and instance.evidence <= memory.tau_e
        )
    else:
        spf_triggered = (
            instance.coverage >= memory.tau_c
            and instance.evidence <= memory.tau_e
            and instance.inspect_count >= memory.tau_n
        )
    instance.spf_triggered = spf_triggered
    return spf_triggered


def update_reliability_for_method(
    memory: SemanticInstanceMemory,
    selected: InstanceState,
    method: str,
    spf_triggered: bool,
) -> float:
    if spf_triggered:
        if method != "ours_no_reliability":
            selected.reliability = memory.lambda_inst * selected.reliability
            if method == "ours_no_category_preserve":
                for instance in memory.all_instances():
                    if (
                        instance.instance_id != selected.instance_id
                        and instance.category == selected.category
                    ):
                        instance.reliability = (
                            memory.lambda_inst * instance.reliability
                        )
        return selected.reliability

    if selected.evidence > memory.tau_e:
        selected.reliability = selected.reliability + memory.alpha * (
            1.0 - selected.reliability
        )
    return selected.reliability


def get_continuation_instance(
    memory: SemanticInstanceMemory,
    active_instance_id: str | None,
) -> InstanceState | None:
    if active_instance_id is None:
        return None

    instance = memory.get_instance(active_instance_id)
    if (
        getattr(instance, "active_inspection", False)
        and instance.coverage < memory.tau_c
        and instance.evidence <= memory.tau_e
    ):
        return instance
    return None


def should_release_active_inspection(
    selected: InstanceState,
    memory: SemanticInstanceMemory,
    method: str,
) -> bool:
    if selected.evidence > memory.tau_e:
        return True
    if method == "ours_no_repeat":
        return selected.coverage >= memory.tau_c
    return (
        selected.coverage >= memory.tau_c
        and selected.inspect_count >= memory.tau_n
    )


def run_decisions(
    episode: dict[str, Any],
    memory: SemanticInstanceMemory,
    inspector: ControlledInspector,
    logger: Any,
    max_decisions: int,
    method: str,
    beta_v: float,
    quiet: bool = False,
    partial_inspection: bool = False,
    poses_per_decision: int = 1,
) -> dict[str, Any]:
    candidates_by_id = {
        candidate["instance_id"]: candidate
        for candidate in episode["candidate_instances"]
    }
    selected_sequence = []
    visit_counts = {
        candidate["instance_id"]: 0 for candidate in episode["candidate_instances"]
    }
    active_instance_id = None
    success = False

    for step in range(max_decisions):
        accessibility = {
            instance.instance_id: 1.0 for instance in memory.all_instances()
        }
        information_gain = {
            instance.instance_id: max(0.5, 1.0 - instance.coverage)
            for instance in memory.all_instances()
        }
        candidate_scores = compute_candidate_scores(
            episode, memory, information_gain, method, visit_counts, beta_v
        )
        top_candidate_scores = candidate_scores[:5]
        selected = None
        selection_reason = "max_utility"
        if partial_inspection and method in OURS_METHODS:
            selected = get_continuation_instance(memory, active_instance_id)
            if selected is not None:
                selection_reason = "continue_active_inspection"
        if selected is None:
            selected = select_instance(
                episode=episode,
                memory=memory,
                accessibility=accessibility,
                information_gain=information_gain,
                step=step,
                method=method,
                visit_counts=visit_counts,
                beta_v=beta_v,
            )
        if selected is None:
            break

        if method == "sp_greedy":
            utility = selected.p_sem
        elif method == "sp_visit_penalty":
            utility = selected.p_sem * math.exp(
                -beta_v * visit_counts[selected.instance_id]
            )
        else:
            utility = memory.compute_utility(
                selected.instance_id,
                accessibility=1.0,
                information_gain=information_gain[selected.instance_id],
            )
        reliability_before = selected.reliability
        visited_viewpoint_ids = None
        decision_pose_limit = None
        if partial_inspection:
            visited_viewpoint_ids = getattr(
                selected, "visited_viewpoint_ids", set()
            )
            decision_pose_limit = poses_per_decision
        observation_info = inspector.inspect(
            episode,
            candidates_by_id[selected.instance_id],
            poses_per_decision=decision_pose_limit,
            visited_viewpoint_ids=visited_viewpoint_ids,
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
        if partial_inspection:
            selected.visited_viewpoint_ids = set(
                observation_info["visited_viewpoint_ids"]
            )
            selected.total_inspection_poses = observation_info[
                "total_inspection_poses"
            ]
            selected.active_inspection = True
            active_instance_id = selected.instance_id
        if method in OURS_METHODS:
            spf_triggered = detect_spf_for_method(
                selected, memory, method
            )
            reliability_after = update_reliability_for_method(
                memory, selected, method, spf_triggered
            )
        else:
            spf_triggered = False
            reliability_after = selected.reliability
        if method == "sp_visit_penalty":
            visit_counts[selected.instance_id] += 1

        success = (
            selected.instance_id == episode["true_support_instance_id"]
            and float(observation_info["evidence"]) == 1.0
        )
        if partial_inspection and method in OURS_METHODS:
            if should_release_active_inspection(selected, memory, method):
                selected.active_inspection = False
                if active_instance_id == selected.instance_id:
                    active_instance_id = None
        selected_sequence.append(selected.instance_id)
        row = {
            "episode_id": episode["episode_id"],
            "method": METHOD_LABELS[method],
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
            "visit_count": visit_counts[selected.instance_id],
            "reliability_before": reliability_before,
            "reliability_after": reliability_after,
            "spf_triggered": spf_triggered,
            "target_visible": bool(observation_info["target_visible"]),
            "success": success,
            "finish_inspection": True,
            "visited_viewpoint_id": observation_info.get(
                "visited_viewpoint_id", ""
            ),
            "visited_viewpoint_ids": sorted(
                observation_info.get("visited_viewpoint_ids", set())
            ),
            "newly_visited_viewpoint_ids": sorted(
                observation_info.get("newly_visited_viewpoint_ids", set())
            ),
            "num_inspection_poses": observation_info[
                "num_inspection_poses"
            ],
            "total_inspection_poses": observation_info[
                "total_inspection_poses"
            ],
            "num_visited_viewpoints": observation_info[
                "num_visited_viewpoints"
            ],
            "active_inspection": bool(
                getattr(selected, "active_inspection", False)
            ),
            "selection_reason": selection_reason,
            "top_candidate_scores": top_candidate_scores,
        }
        logger.log_step(row)
        if not quiet:
            print(
                f"step={step} selected_alias={selected.alias} "
                f"selected_category={selected.category} "
                f"p_sem={selected.p_sem:.4f} "
                f"coverage={selected.coverage:.4f} "
                f"evidence={selected.evidence:.4f} "
                f"inspect_count={selected.inspect_count} "
                f"visit_count={visit_counts[selected.instance_id]} "
                f"selection_reason={selection_reason} "
                f"reliability_before={reliability_before:.4f} "
                f"reliability_after={reliability_after:.4f} "
                f"utility={utility:.4f} "
                f"spf_triggered={spf_triggered} success={success}"
            )
            print("top candidates:")
            for rank, score in enumerate(candidate_scores[:3], start=1):
                print(
                    f"{rank}. {score['alias']} U={score['utility']:.3f} "
                    f"R={score['reliability']:.3f} "
                    f"C={score['coverage']:.3f} E={score['evidence']:.3f}"
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

    final_memory = [
        {
            "instance_id": instance.instance_id,
            "alias": instance.alias,
            "category": instance.category,
            "p_sem": instance.p_sem,
            "reliability": instance.reliability,
            "coverage": instance.coverage,
            "evidence": instance.evidence,
            "inspect_count": instance.inspect_count,
        }
        for instance in memory.all_instances()
    ]

    return {
        "episode_id": episode["episode_id"],
        "method": METHOD_LABELS[method],
        "target_category": episode["target_category"],
        "episode_type": episode["episode_type"],
        "success": success,
        "num_steps": len(selected_sequence),
        "selected_sequence": selected_sequence,
        "wrong_instance_id": wrong_instance_id,
        "true_support_instance_id": episode["true_support_instance_id"],
        "final_reliability_wrong": final_reliability_wrong,
        "reliability_drop_wrong": reliability_drop_wrong,
        "final_memory": final_memory,
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
        "--method",
        choices=METHOD_CHOICES,
        default="ours",
    )
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
    parser.add_argument("--beta-v", type=float, default=0.7)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--partial-inspection", action="store_true")
    parser.add_argument("--poses-per-decision", type=int, default=1)
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
    logger = EpisodeLogger(
        args.output.parent, args.output.name, METHOD_LABELS[args.method]
    )
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
            method=args.method,
            beta_v=args.beta_v,
            quiet=args.quiet,
            partial_inspection=args.partial_inspection,
            poses_per_decision=args.poses_per_decision,
        )
        logger.save_summary(summary)
        if args.quiet:
            print(
                f"method={args.method} episode_id={episode['episode_id']} "
                f"success={summary['success']} num_steps={summary['num_steps']}"
            )
        else:
            print(f"step log: {logger.step_log_path}")
            print(f"episode summary: {logger.summary_path}")
    finally:
        if controller is not None:
            controller.stop()
        logger.close()


if __name__ == "__main__":
    main()

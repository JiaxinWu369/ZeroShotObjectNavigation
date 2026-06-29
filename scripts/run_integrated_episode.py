"""Run a minimal metadata-perception integrated ObjectNav episode."""

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
from iac_zson.navigation.grid_planner import GridPlanner
from iac_zson.planning.controlled_inspector import ControlledInspector


METHOD_LABELS = {
    "ours": "Ours",
    "sp_greedy": "SP-Greedy",
    "sp_visit_penalty": "SP+VisitPenalty",
}


def load_episode(path: Path, episode_index: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        episodes = [json.loads(line) for line in file if line.strip()]
    return episodes[episode_index]


def initialize_memory(
    episode: dict[str, Any],
    tau_c: float,
    tau_e: float,
    tau_n: int,
) -> SemanticInstanceMemory:
    memory = SemanticInstanceMemory(
        tau_c=tau_c,
        tau_e=tau_e,
        tau_n=tau_n,
        lambda_inst=0.35,
        alpha=0.25,
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
        )
        instance.visited_viewpoint_ids = set()
        instance.active_inspection = False
        memory.add_instance(instance)
    return memory


def get_agent_pose(controller: Any) -> dict[str, Any]:
    agent = controller.last_event.metadata.get("agent", {})
    return {
        "position": dict(agent.get("position", {})),
        "rotation": dict(agent.get("rotation", {})),
        "cameraHorizon": agent.get("cameraHorizon"),
    }


def teleport_to_start(controller: Any, episode: dict[str, Any]) -> None:
    start_pose = episode.get("initial_pose") or episode.get("start_pose")
    if not start_pose:
        return
    position = start_pose.get("position", {})
    rotation = start_pose.get("rotation", {})
    controller.step(
        action="TeleportFull",
        x=position.get("x", 0.0),
        y=position.get("y", 0.0),
        z=position.get("z", 0.0),
        rotation={
            "x": rotation.get("x", 0.0),
            "y": rotation.get("y", 0.0),
            "z": rotation.get("z", 0.0),
        },
        horizon=start_pose.get("horizon", 0.0),
        standing=start_pose.get("standing", True),
    )


def select_instance(
    episode: dict[str, Any],
    memory: SemanticInstanceMemory,
    method: str,
    visit_counts: dict[str, int],
    beta_v: float,
    decision_step: int,
) -> InstanceState | None:
    instances = memory.all_instances()
    if not instances:
        return None

    if method == "sp_greedy":
        return max(instances, key=lambda instance: instance.p_sem)

    if method == "sp_visit_penalty":
        return max(
            instances,
            key=lambda instance: instance.p_sem
            * math.exp(-beta_v * visit_counts.get(instance.instance_id, 0)),
        )

    information_gain = {
        instance.instance_id: max(0.5, 1.0 - instance.coverage)
        for instance in instances
    }
    accessibility = {instance.instance_id: 1.0 for instance in instances}
    return memory.select_best_instance(
        accessibility, information_gain, step=decision_step
    )


def should_continue_active_instance(
    memory: SemanticInstanceMemory,
    active_instance_id: str | None,
) -> InstanceState | None:
    if active_instance_id is None:
        return None
    instance = memory.get_instance(active_instance_id)
    if (
        getattr(instance, "active_inspection", False)
        and instance.evidence <= memory.tau_e
        and (
            instance.coverage < memory.tau_c
            or instance.inspect_count < memory.tau_n
        )
    ):
        return instance
    return None


def compute_utility(
    memory: SemanticInstanceMemory,
    instance: InstanceState,
    method: str,
    visit_counts: dict[str, int],
    beta_v: float,
) -> float:
    if method == "sp_greedy":
        return instance.p_sem
    if method == "sp_visit_penalty":
        return instance.p_sem * math.exp(
            -beta_v * visit_counts.get(instance.instance_id, 0)
        )
    return memory.compute_utility(
        instance.instance_id,
        accessibility=1.0,
        information_gain=max(0.5, 1.0 - instance.coverage),
    )


def update_memory_after_inspection(
    memory: SemanticInstanceMemory,
    instance: InstanceState,
    observation: dict[str, Any],
    method: str,
) -> tuple[bool, float]:
    memory.update_coverage(
        instance.instance_id,
        max(instance.coverage, float(observation.get("coverage", 0.0))),
    )
    memory.update_evidence(
        instance.instance_id,
        max(instance.evidence, float(observation.get("evidence", 0.0))),
    )
    memory.finish_inspection(
        instance.instance_id,
        observation.get("visited_viewpoint_id"),
    )
    if hasattr(instance, "visited_viewpoint_ids"):
        instance.visited_viewpoint_ids = set(
            observation.get("visited_viewpoint_ids", set())
        )

    if method == "ours":
        spf_triggered = (
            instance.coverage >= memory.tau_c
            and instance.inspect_count >= memory.tau_n
            and instance.evidence <= memory.tau_e
        )
        instance.spf_triggered = spf_triggered
        if spf_triggered:
            instance.reliability = memory.lambda_inst * instance.reliability
        elif instance.evidence > memory.tau_e:
            instance.reliability = instance.reliability + memory.alpha * (
                1.0 - instance.reliability
            )
        reliability = instance.reliability
    else:
        spf_triggered = False
        reliability = instance.reliability
    return spf_triggered, reliability


def log_row(
    logger: EpisodeLogger,
    episode: dict[str, Any],
    method: str,
    global_step: int,
    decision_step: int,
    selected: InstanceState,
    action: str,
    agent_pose: dict[str, Any],
    path_length: float,
    success: bool,
    utility: float,
    spf_triggered: bool = False,
    action_type: str = "physical",
    simulator_executed: bool = True,
    selection_reason: str = "max_utility",
) -> None:
    logger.log_step(
        {
            "episode_id": episode["episode_id"],
            "method": METHOD_LABELS[method],
            "global_step": global_step,
            "decision_step": decision_step,
            "selected_instance": selected.instance_id,
            "selected_instance_id": selected.instance_id,
            "selected_instance_alias": selected.alias,
            "selected_instance_category": selected.category,
            "action": action,
            "action_type": action_type,
            "simulator_executed": simulator_executed,
            "agent_pose": agent_pose,
            "coverage": selected.coverage,
            "evidence": selected.evidence,
            "inspect_count": selected.inspect_count,
            "reliability": selected.reliability,
            "utility": utility,
            "spf_triggered": spf_triggered,
            "selection_reason": selection_reason,
            "path_length": path_length,
            "success": success,
        }
    )


def run_episode(
    episode: dict[str, Any],
    controller: Any,
    logger: EpisodeLogger,
    method: str,
    max_steps: int,
    max_decisions: int,
    tau_c: float,
    tau_e: float,
    tau_n: int,
    partial_inspection: bool,
    poses_per_decision: int,
    beta_v: float,
) -> dict[str, Any]:
    memory = initialize_memory(episode, tau_c=tau_c, tau_e=tau_e, tau_n=tau_n)
    candidates_by_id = {
        candidate["instance_id"]: candidate
        for candidate in episode["candidate_instances"]
    }
    visit_counts = {instance_id: 0 for instance_id in candidates_by_id}
    planner = GridPlanner(controller)
    inspector = ControlledInspector(controller)

    global_step = 0
    path_length = 0.0
    selected_sequence: list[str] = []
    success = False
    stopped = False
    terminal_action = None
    active_instance_id = None

    for decision_step in range(max_decisions):
        if global_step >= max_steps:
            break

        selection_reason = "max_utility"
        selected = None
        if method == "ours" and partial_inspection:
            selected = should_continue_active_instance(memory, active_instance_id)
            if selected is not None:
                selection_reason = "continue_active_inspection"
        if selected is None:
            selected = select_instance(
                episode,
                memory,
                method,
                visit_counts,
                beta_v,
                decision_step,
            )
        if selected is None:
            break
        selected_sequence.append(selected.instance_id)
        utility = compute_utility(memory, selected, method, visit_counts, beta_v)

        agent_pose = get_agent_pose(controller)
        selected_candidate = candidates_by_id[selected.instance_id]
        path = planner.shortest_path(
            agent_pose["position"], selected_candidate["position"]
        )
        actions = planner.path_to_actions(agent_pose, path)

        for action in actions:
            if global_step >= max_steps:
                break
            event = controller.step(action=action)
            if action == "MoveAhead" and event.metadata.get(
                "lastActionSuccess", True
            ):
                path_length += planner.grid_size
            global_step += 1
            log_row(
                logger,
                episode,
                method,
                global_step,
                decision_step,
                selected,
                action,
                get_agent_pose(controller),
                path_length,
                success,
                utility,
                selection_reason=selection_reason,
            )

        if global_step >= max_steps:
            break

        observation = inspector.inspect(
            episode,
            selected_candidate,
            poses_per_decision=poses_per_decision if partial_inspection else None,
            visited_viewpoint_ids=getattr(selected, "visited_viewpoint_ids", set()),
        )
        spf_triggered, reliability = update_memory_after_inspection(
            memory, selected, observation, method
        )
        selected.reliability = reliability
        if method == "ours" and partial_inspection:
            selected.active_inspection = True
            active_instance_id = selected.instance_id
            if spf_triggered or selected.evidence > tau_e:
                selected.active_inspection = False
                active_instance_id = None
        if method == "sp_visit_penalty":
            visit_counts[selected.instance_id] += 1

        global_step += 1
        success = float(observation.get("evidence", 0.0)) > tau_e
        log_row(
            logger,
            episode,
            method,
            global_step,
            decision_step,
            selected,
            "Inspect",
            get_agent_pose(controller),
            path_length,
            success,
            utility,
            spf_triggered=spf_triggered,
            selection_reason=selection_reason,
        )

        if success:
            log_row(
                logger,
                episode,
                method,
                global_step,
                decision_step,
                selected,
                "Stop",
                get_agent_pose(controller),
                path_length,
                success,
                utility,
                spf_triggered=spf_triggered,
                action_type="virtual",
                simulator_executed=False,
                selection_reason=selection_reason,
            )
            stopped = True
            terminal_action = "Stop"
            break

    wrong_id = episode.get("wrong_instance_id")
    true_id = episode.get("true_support_instance_id")
    final_wrong = (
        memory.get_instance(wrong_id).reliability
        if wrong_id in candidates_by_id
        else None
    )
    final_true = (
        memory.get_instance(true_id).reliability
        if true_id in candidates_by_id
        else None
    )
    return {
        "episode_id": episode["episode_id"],
        "method": METHOD_LABELS[method],
        "success": success,
        "stopped": stopped,
        "terminal_action": terminal_action,
        "num_actions": global_step,
        "num_decisions": len(selected_sequence),
        "path_length": path_length,
        "selected_sequence": selected_sequence,
        "wrong_instance_id": wrong_id,
        "true_support_instance_id": true_id,
        "final_reliability_wrong": final_wrong,
        "final_reliability_true_support": final_true,
        "reliability_drop_wrong": (
            None if final_wrong is None else 1.0 - final_wrong
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes",
        type=Path,
        default=PROJECT_ROOT / "data" / "episodes" / "debug_episodes.jsonl",
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--method",
        choices=("ours", "sp_greedy", "sp_visit_penalty"),
        default="ours",
    )
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--max-decisions", type=int, default=30)
    parser.add_argument("--tau-c", type=float, default=0.75)
    parser.add_argument("--tau-n", type=int, default=2)
    parser.add_argument("--partial-inspection", action="store_true")
    parser.add_argument("--poses-per-decision", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "logs" / "integrated_episode",
    )
    parser.add_argument("--beta-v", type=float, default=0.7)
    args = parser.parse_args()

    episode = load_episode(args.episodes, args.episode_index)
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
        teleport_to_start(controller, episode)
        summary = run_episode(
            episode=episode,
            controller=controller,
            logger=logger,
            method=args.method,
            max_steps=args.max_steps,
            max_decisions=args.max_decisions,
            tau_c=args.tau_c,
            tau_e=0.1,
            tau_n=args.tau_n,
            partial_inspection=args.partial_inspection,
            poses_per_decision=args.poses_per_decision,
            beta_v=args.beta_v,
        )
        logger.save_summary(summary)
        print(f"success={summary['success']}")
        print(f"num_actions={summary['num_actions']}")
        print(f"path_length={summary['path_length']:.3f}")
        print(f"step log: {logger.step_log_path}")
        print(f"episode summary: {logger.summary_path}")
    finally:
        if controller is not None:
            controller.stop()
        logger.close()


if __name__ == "__main__":
    main()

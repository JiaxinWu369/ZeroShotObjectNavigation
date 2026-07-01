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
from iac_zson.planning.inspection_points import compute_yaw_to_target


METHOD_LABELS = {
    "ours": "Ours",
    "sp_greedy": "SP-Greedy",
    "sp_visit_penalty": "SP+VisitPenalty",
}
NAVIGATION_FORCED_SWITCH_REASONS = {"navigation_failed", "accessibility_zero"}


class FrameSaver:
    """Optional RGB frame writer for demonstration runs."""

    def __init__(self, output_dir: Path, save_every: int = 1) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = max(1, save_every)
        self.frame_index = 0

    def maybe_save(self, frame: Any, global_step: int) -> str | None:
        if global_step % self.save_every != 0:
            return None
        self.frame_index += 1
        output_path = self.output_dir / f"frame_{self.frame_index:06d}.png"
        save_rgb_frame(frame, output_path)
        return str(output_path)


def save_rgb_frame(frame: Any, output_path: Path) -> None:
    """Save an RGB frame using PIL first, then imageio."""
    pil_error: Exception | None = None
    try:
        from PIL import Image

        Image.fromarray(frame).save(output_path)
        return
    except ImportError as error:
        pil_error = error
    except Exception as error:
        pil_error = error

    imageio_error: Exception | None = None
    try:
        import imageio.v2 as imageio

        imageio.imwrite(output_path, frame)
        return
    except ImportError as error:
        imageio_error = error
    except Exception as error:
        imageio_error = error

    raise RuntimeError(
        "Failed to save RGB frame. Install or fix image writers with: "
        "pip install pillow imageio. "
        f"PIL error: {pil_error}; imageio error: {imageio_error}"
    )


def to_jsonable_number(value: Any) -> float | int | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_jsonable_bbox(bbox: Any) -> list[float | int] | None:
    if bbox is None:
        return None
    if hasattr(bbox, "tolist"):
        bbox = bbox.tolist()
    try:
        values = list(bbox)
    except TypeError:
        return None
    jsonable_values = [to_jsonable_number(value) for value in values]
    if any(value is None for value in jsonable_values):
        return None
    return [value for value in jsonable_values if value is not None]


def to_jsonable_position(position: dict[str, Any] | None) -> dict[str, float] | None:
    if not position:
        return None
    jsonable_position = {}
    for axis in ("x", "y", "z"):
        value = to_jsonable_number(position.get(axis))
        if value is not None:
            jsonable_position[axis] = float(value)
    return jsonable_position or None


def extract_target_frame_info(event: Any, episode: dict[str, Any]) -> dict[str, Any]:
    metadata = getattr(event, "metadata", {}) or {}
    objects = metadata.get("objects", []) or []
    target_category = episode.get("target_category")
    preferred_target_id = episode.get("target_object_id")

    target_object = None
    if preferred_target_id:
        target_object = next(
            (
                obj
                for obj in objects
                if obj.get("objectId") == preferred_target_id
            ),
            None,
        )
    if target_object is None:
        target_object = next(
            (
                obj
                for obj in objects
                if obj.get("objectType") == target_category
            ),
            None,
        )

    target_object_id = (
        target_object.get("objectId")
        if target_object is not None
        else preferred_target_id
    )
    detections_2d = getattr(event, "instance_detections2D", {}) or {}
    target_bbox_2d = None
    if target_object_id in detections_2d:
        target_bbox_2d = to_jsonable_bbox(detections_2d[target_object_id])

    return {
        "target_object_id": target_object_id,
        "target_visible": bool(target_object.get("visible", False))
        if target_object is not None
        else False,
        "target_position": to_jsonable_position(target_object.get("position"))
        if target_object is not None
        else None,
        "target_bbox_2d": target_bbox_2d,
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
        instance.failed_viewpoint_ids = set()
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


def safe_step(controller: Any, action: str, **kwargs: Any) -> tuple[bool, Any, str]:
    try:
        event = controller.step(action=action, **kwargs)
    except (ValueError, TimeoutError, RuntimeError) as error:
        return False, None, str(error)
    return True, event, ""


def teleport_to_start(controller: Any, episode: dict[str, Any]) -> None:
    start_pose = episode.get("initial_pose") or episode.get("start_pose")
    if not start_pose:
        return
    position = start_pose.get("position", {})
    rotation = start_pose.get("rotation", {})
    safe_step(
        controller,
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
    accessibility: dict[str, float],
) -> InstanceState | None:
    instances = memory.all_instances()
    if not instances:
        return None
    if not any(
        accessibility.get(instance.instance_id, 1.0) > 0.0
        for instance in instances
    ):
        return None

    if method == "sp_greedy":
        scored = [
            (instance.p_sem * accessibility.get(instance.instance_id, 1.0), instance)
            for instance in instances
        ]
        best_score, best_instance = max(scored, key=lambda item: item[0])
        return best_instance if best_score > 0.0 else None

    if method == "sp_visit_penalty":
        scored = [
            (
                instance.p_sem
                * accessibility.get(instance.instance_id, 1.0)
                * math.exp(-beta_v * visit_counts.get(instance.instance_id, 0)),
                instance,
            )
            for instance in instances
        ]
        best_score, best_instance = max(scored, key=lambda item: item[0])
        return best_instance if best_score > 0.0 else None

    information_gain = {
        instance.instance_id: max(0.5, 1.0 - instance.coverage)
        for instance in instances
    }
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
    accessibility: dict[str, float],
) -> float:
    accessibility_value = accessibility.get(instance.instance_id, 1.0)
    if method == "sp_greedy":
        return instance.p_sem * accessibility_value
    if method == "sp_visit_penalty":
        return instance.p_sem * accessibility_value * math.exp(
            -beta_v * visit_counts.get(instance.instance_id, 0)
        )
    return memory.compute_utility(
        instance.instance_id,
        accessibility=accessibility_value,
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


def is_agent_in_inspection_region(
    agent_pose: dict[str, Any],
    inspector: ControlledInspector,
    selected_candidate: dict[str, Any],
    tolerance: float,
) -> bool:
    agent_position = agent_pose.get("position", {})
    try:
        agent_x = float(agent_position["x"])
        agent_z = float(agent_position["z"])
    except (KeyError, TypeError, ValueError):
        return False

    for pose in inspector._get_inspection_poses(selected_candidate):
        dx = agent_x - float(pose["x"])
        dz = agent_z - float(pose["z"])
        if math.sqrt(dx * dx + dz * dz) <= tolerance:
            return True
    return False


def horizontal_distance_to_position(
    agent_pose: dict[str, Any],
    target_position: dict[str, Any],
) -> float:
    agent_position = agent_pose.get("position", {})
    try:
        dx = float(agent_position["x"]) - float(target_position["x"])
        dz = float(agent_position["z"]) - float(target_position["z"])
    except (KeyError, TypeError, ValueError):
        return float("inf")
    return math.sqrt(dx * dx + dz * dz)


def rotate_actions_to_yaw(current_yaw: float, desired_yaw: float) -> list[str]:
    current = int(round(current_yaw / 90.0) * 90) % 360
    desired = int(round(desired_yaw / 90.0) * 90) % 360
    diff = (desired - current) % 360
    if diff == 0:
        return []
    if diff == 90:
        return ["RotateRight"]
    if diff == 180:
        return ["RotateRight", "RotateRight"]
    if diff == 270:
        return ["RotateLeft"]
    return []


def read_target_view_info(
    metadata: dict[str, Any],
    episode: dict[str, Any],
) -> tuple[bool, float | None, str | None]:
    target_category = episode.get("target_category")
    preferred_target_id = episode.get("target_object_id")
    best_distance = None
    best_object_id = None

    for obj in metadata.get("objects", []) or []:
        if obj.get("objectType") != target_category:
            continue
        if preferred_target_id and obj.get("objectId") != preferred_target_id:
            continue
        if not obj.get("visible", False):
            continue
        distance = obj.get("distance")
        if distance is None:
            position = obj.get("position") or {}
            agent = metadata.get("agent", {}).get("position", {})
            try:
                distance = math.sqrt(
                    (float(position["x"]) - float(agent["x"])) ** 2
                    + (float(position["y"]) - float(agent.get("y", 0.0))) ** 2
                    + (float(position["z"]) - float(agent["z"])) ** 2
                )
            except (KeyError, TypeError, ValueError):
                distance = None
        if distance is None:
            continue
        distance = float(distance)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_object_id = obj.get("objectId")

    return best_distance is not None, best_distance, best_object_id


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
    frame_path: str | None = None,
    phase: str = "navigation_action",
    reliability_before: float | None = None,
    reliability_after: float | None = None,
    target_frame_info: dict[str, Any] | None = None,
    navigation_skipped_for_active_inspection: bool = False,
    active_release_reason: str | None = None,
    observation_info: dict[str, Any] | None = None,
    success_info: dict[str, Any] | None = None,
    switch_reason: str | None = None,
    navigation_forced_switch: bool = False,
) -> None:
    row = {
        "episode_id": episode["episode_id"],
        "method": METHOD_LABELS[method],
        "phase": phase,
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
        "navigation_skipped_for_active_inspection": (
            navigation_skipped_for_active_inspection
        ),
        "path_length": path_length,
        "success": success,
    }
    if switch_reason is not None:
        row["switch_reason"] = switch_reason
        row["navigation_forced_switch"] = navigation_forced_switch
    if frame_path is not None:
        row["frame_path"] = frame_path
        target_frame_info = target_frame_info or {}
        row["target_object_id"] = target_frame_info.get("target_object_id")
        row["target_visible"] = target_frame_info.get("target_visible")
        row["target_position"] = target_frame_info.get("target_position")
        row["target_bbox_2d"] = target_frame_info.get("target_bbox_2d")
    if reliability_before is not None:
        row["reliability_before"] = reliability_before
    if reliability_after is not None:
        row["reliability_after"] = reliability_after
    if active_release_reason is not None:
        row["active_release_reason"] = active_release_reason
    if observation_info is not None:
        if "target_visible" in observation_info:
            row["target_visible"] = observation_info.get("target_visible")
        if "target_distance" in observation_info:
            row["target_distance"] = observation_info.get("target_distance")
        if "evidence" in observation_info:
            row["evidence"] = observation_info.get("evidence")
        if "evidence_visible" in observation_info:
            row["evidence_visible"] = observation_info.get("evidence_visible")
        if "evidence_support" in observation_info:
            row["evidence_support"] = observation_info.get("evidence_support")
    if success_info is not None:
        row.update(success_info)
    logger.log_step(row)


def log_decision_update(
    logger: EpisodeLogger,
    episode: dict[str, Any],
    method: str,
    global_step: int,
    decision_step: int,
    selected: InstanceState,
    path_length: float,
    success: bool,
    utility: float,
    spf_triggered: bool,
    selection_reason: str,
    reliability_before: float,
    reliability_after: float,
    representative_frame_path: str | None,
    navigation_skipped_for_active_inspection: bool = False,
    active_release_reason: str | None = None,
    success_info: dict[str, Any] | None = None,
    switch_reason: str | None = None,
    navigation_forced_switch: bool = False,
) -> None:
    row = {
            "episode_id": episode["episode_id"],
            "method": METHOD_LABELS[method],
            "phase": "decision_update",
            "global_step": global_step,
            "decision_step": decision_step,
            "selected_instance": selected.instance_id,
            "selected_instance_id": selected.instance_id,
            "selected_instance_alias": selected.alias,
            "selected_instance_category": selected.category,
            "coverage": selected.coverage,
            "evidence": selected.evidence,
            "inspect_count": selected.inspect_count,
            "reliability": selected.reliability,
            "reliability_before": reliability_before,
            "reliability_after": reliability_after,
            "utility": utility,
            "spf_triggered": spf_triggered,
            "selection_reason": selection_reason,
            "navigation_skipped_for_active_inspection": (
                navigation_skipped_for_active_inspection
            ),
            "path_length": path_length,
            "success": success,
            "representative_frame_path": representative_frame_path,
            "active_release_reason": active_release_reason,
        }
    if switch_reason is not None:
        row["switch_reason"] = switch_reason
        row["navigation_forced_switch"] = navigation_forced_switch
    if success_info is not None:
        row.update(success_info)
    logger.log_step(row)


def make_success_info(
    observation: dict[str, Any],
    selected: InstanceState,
    true_support_instance_id: str | None,
    success_source: str,
    global_step: int,
) -> dict[str, Any]:
    return {
        "success_source": success_source,
        "success_trigger_action": (
            observation.get("success_trigger_action") or "Inspect"
        ),
        "success_trigger_step": (
            observation.get("success_trigger_step") or global_step
        ),
        "success_selected_instance_id": selected.instance_id,
        "success_selected_instance_alias": selected.alias,
        "success_target_visible": bool(observation.get("target_visible", False)),
        "success_evidence": float(observation.get("evidence", 0.0)),
        "success_target_distance": observation.get("target_distance"),
        "success_is_true_support": selected.instance_id == true_support_instance_id,
    }


def log_action_failed(
    logger: EpisodeLogger,
    episode: dict[str, Any],
    method: str,
    global_step: int,
    decision_step: int,
    selected: InstanceState,
    action: str,
    error_message: str,
    agent_pose_before: dict[str, Any],
    agent_pose_after: dict[str, Any],
    path_length: float,
    selection_reason: str,
    frame_path: str | None = None,
    target_frame_info: dict[str, Any] | None = None,
) -> None:
    row = {
        "episode_id": episode["episode_id"],
        "method": METHOD_LABELS[method],
        "phase": "action_failed",
        "global_step": global_step,
        "decision_step": decision_step,
        "selected_instance": selected.instance_id,
        "selected_instance_id": selected.instance_id,
        "selected_instance_alias": selected.alias,
        "selected_instance_category": selected.category,
        "action": action,
        "action_type": "physical",
        "simulator_executed": True,
        "action_failed": True,
        "failed_action": action,
        "lastActionSuccess": False,
        "error_message": error_message,
        "errorMessage": error_message,
        "agent_pose_before": agent_pose_before,
        "agent_pose_after": agent_pose_after,
        "path_length": path_length,
        "selection_reason": selection_reason,
        "success": False,
    }
    if frame_path is not None:
        row["frame_path"] = frame_path
        target_frame_info = target_frame_info or {}
        row["target_object_id"] = target_frame_info.get("target_object_id")
        row["target_visible"] = target_frame_info.get("target_visible")
        row["target_position"] = target_frame_info.get("target_position")
        row["target_bbox_2d"] = target_frame_info.get("target_bbox_2d")
    logger.log_step(row)


def log_navigation_failed_for_instance(
    logger: EpisodeLogger,
    episode: dict[str, Any],
    method: str,
    global_step: int,
    decision_step: int,
    selected: InstanceState,
    path_length: float,
    selection_reason: str,
    error_message: str,
    failed_viewpoint_ids: set[str],
    active_release_reason: str | None = None,
) -> None:
    logger.log_step(
        {
            "episode_id": episode["episode_id"],
            "method": METHOD_LABELS[method],
            "phase": "navigation_failed_for_instance",
            "global_step": global_step,
            "decision_step": decision_step,
            "selected_instance": selected.instance_id,
            "selected_instance_id": selected.instance_id,
            "selected_instance_alias": selected.alias,
            "selected_instance_category": selected.category,
            "action": "NavigationFailed",
            "action_type": "virtual",
            "simulator_executed": False,
            "errorMessage": error_message,
            "path_length": path_length,
            "selection_reason": selection_reason,
            "switch_reason": "navigation_failed",
            "navigation_forced_switch": True,
            "accessibility": 0.0,
            "failed_viewpoint_ids": sorted(failed_viewpoint_ids),
            "navigation_failed_for_instance": True,
            "active_release_reason": active_release_reason,
            "success": False,
        }
    )


def inspection_pose_items_for_navigation(
    inspector: ControlledInspector,
    selected_candidate: dict[str, Any],
    visited_viewpoint_ids: set[str],
    failed_viewpoint_ids: set[str],
    max_viewpoint_retries: int,
) -> list[tuple[str, dict[str, float]]]:
    selected_instance_id = selected_candidate["instance_id"]
    pose_items = list(enumerate(inspector._get_inspection_poses(selected_candidate)))
    candidate_pose_items = [
        (inspector._viewpoint_id(selected_instance_id, pose_index), pose)
        for pose_index, pose in pose_items
        if (
            inspector._viewpoint_id(selected_instance_id, pose_index)
            not in visited_viewpoint_ids
            and inspector._viewpoint_id(selected_instance_id, pose_index)
            not in failed_viewpoint_ids
        )
    ]
    return candidate_pose_items[:max_viewpoint_retries]


def all_inspection_viewpoint_ids(
    inspector: ControlledInspector,
    selected_candidate: dict[str, Any],
) -> set[str]:
    selected_instance_id = selected_candidate["instance_id"]
    return {
        inspector._viewpoint_id(selected_instance_id, pose_index)
        for pose_index, _ in enumerate(
            inspector._get_inspection_poses(selected_candidate)
        )
    }


def navigate_to_pose(
    controller: Any,
    planner: GridPlanner,
    logger: EpisodeLogger,
    episode: dict[str, Any],
    method: str,
    selected: InstanceState,
    decision_step: int,
    goal_position: dict[str, float],
    max_steps: int,
    global_step: int,
    path_length: float,
    success: bool,
    utility: float,
    selection_reason: str,
    frame_saver: FrameSaver | None,
    navigation_skipped_for_active_inspection: bool,
) -> tuple[bool, int, float, str | None]:
    agent_pose = get_agent_pose(controller)
    path = planner.shortest_path(agent_pose["position"], goal_position)
    if not path:
        return False, global_step, path_length, "planner_failed"

    actions = planner.path_to_actions(agent_pose, path)
    if len(path) >= 2 and not actions:
        return False, global_step, path_length, "planner_failed"

    action_index = 0
    replan_attempted = False
    while action_index < len(actions):
        if global_step >= max_steps:
            return False, global_step, path_length, "max_steps_reached"

        action = actions[action_index]
        agent_pose_before = get_agent_pose(controller)
        ok, event, safe_step_error = safe_step(controller, action=action)
        global_step += 1
        frame_path = (
            frame_saver.maybe_save(controller.last_event.frame, global_step)
            if frame_saver is not None and ok
            else None
        )
        target_frame_info = (
            extract_target_frame_info(controller.last_event, episode)
            if frame_path is not None
            else None
        )
        last_action_success = (
            ok and event.metadata.get("lastActionSuccess", True)
        )
        agent_pose_after = get_agent_pose(controller)

        if not last_action_success:
            error_message = (
                safe_step_error
                or (event.metadata.get("errorMessage") if event is not None else "")
                or action
            )
            log_action_failed(
                logger,
                episode,
                method,
                global_step,
                decision_step,
                selected,
                action,
                error_message,
                agent_pose_before,
                agent_pose_after,
                path_length,
                selection_reason,
                frame_path=frame_path,
                target_frame_info=target_frame_info,
            )
            if replan_attempted:
                return False, global_step, path_length, error_message

            replan_attempted = True
            replanned_pose = get_agent_pose(controller)
            replanned_path = planner.shortest_path(
                replanned_pose["position"],
                goal_position,
            )
            if not replanned_path:
                return False, global_step, path_length, error_message

            actions = planner.path_to_actions(replanned_pose, replanned_path)
            if len(replanned_path) >= 2 and not actions:
                return False, global_step, path_length, error_message
            action_index = 0
            continue

        if action == "MoveAhead":
            path_length += planner.grid_size
        log_row(
            logger,
            episode,
            method,
            global_step,
            decision_step,
            selected,
            action,
            agent_pose_after,
            path_length,
            success,
            utility,
            selection_reason=selection_reason,
            frame_path=frame_path,
            phase="navigation_action",
            target_frame_info=target_frame_info,
            navigation_skipped_for_active_inspection=(
                navigation_skipped_for_active_inspection
            ),
        )
        action_index += 1

    return True, global_step, path_length, None


def perform_local_scan(
    controller: Any,
    logger: EpisodeLogger,
    episode: dict[str, Any],
    method: str,
    selected: InstanceState,
    selected_candidate: dict[str, Any],
    inspector: ControlledInspector,
    decision_step: int,
    global_step: int,
    path_length: float,
    success: bool,
    utility: float,
    selection_reason: str,
    frame_saver: FrameSaver | None,
    tau_e: float,
    success_distance: float,
    visited_viewpoint_id: str | None = None,
) -> tuple[dict[str, Any], int, str | None]:
    target_category = episode["target_category"]
    selected_instance_id = selected.instance_id
    target_visible = False
    target_visible_within_success_distance = False
    min_target_distance = None
    evidence_found = False
    representative_frame_path = None
    target_checks: list[dict[str, Any]] = []
    success_trigger_action = None
    success_trigger_step = None

    def read_current_view(view_label: str) -> None:
        nonlocal target_visible, target_visible_within_success_distance
        nonlocal min_target_distance, evidence_found
        nonlocal success_trigger_action, success_trigger_step
        visible_now, evidence_now = inspector._read_observation(
            controller.last_event.metadata,
            target_category,
            selected_instance_id,
        )
        target_visible_now, target_distance_now, _ = read_target_view_info(
            controller.last_event.metadata,
            episode,
        )
        target_checks.append(
            {
                "view": view_label,
                "target_visible": bool(target_visible_now),
                "target_distance": target_distance_now,
            }
        )
        target_visible = target_visible or visible_now
        target_visible = target_visible or target_visible_now
        if target_distance_now is not None:
            min_target_distance = (
                target_distance_now
                if min_target_distance is None
                else min(min_target_distance, target_distance_now)
            )
            if target_distance_now <= success_distance:
                target_visible_within_success_distance = True
        evidence_found = evidence_found or evidence_now
        if (
            success_trigger_action is None
            and (evidence_now or target_visible_within_success_distance)
        ):
            success_trigger_action = view_label
            success_trigger_step = global_step

    def execute_scan_action(action: str, should_check: bool) -> bool:
        nonlocal global_step, representative_frame_path
        agent_pose_before = get_agent_pose(controller)
        ok, event, safe_step_error = safe_step(controller, action=action)
        global_step += 1
        frame_path = (
            frame_saver.maybe_save(controller.last_event.frame, global_step)
            if frame_saver is not None and ok
            else None
        )
        if frame_path is not None:
            representative_frame_path = frame_path
        target_frame_info = (
            extract_target_frame_info(controller.last_event, episode)
            if frame_path is not None
            else None
        )
        agent_pose_after = get_agent_pose(controller)
        last_action_success = (
            ok and event.metadata.get("lastActionSuccess", True)
        )
        if not last_action_success:
            error_message = (
                safe_step_error
                or (event.metadata.get("errorMessage") if event is not None else "")
                or action
            )
            log_action_failed(
                logger,
                episode,
                method,
                global_step,
                decision_step,
                selected,
                action,
                error_message,
                agent_pose_before,
                agent_pose_after,
                path_length,
                selection_reason,
                frame_path=frame_path,
                target_frame_info=target_frame_info,
            )
            return False

        if should_check:
            read_current_view(action)
        log_row(
            logger,
            episode,
            method,
            global_step,
            decision_step,
            selected,
            action,
            agent_pose_after,
            path_length,
            success or target_visible_within_success_distance,
            utility,
            selection_reason=selection_reason,
            frame_path=frame_path,
            phase="inspection_action",
            target_frame_info=target_frame_info,
            observation_info={
                "target_visible": target_visible,
                "target_distance": min_target_distance,
                "evidence": 1.0 if evidence_found else 0.0,
                "evidence_visible": 1.0 if target_visible else 0.0,
                "evidence_support": (
                    1.0
                    if selected_instance_id
                    == episode.get("true_support_instance_id")
                    else 0.0
                ),
            },
        )
        return True

    agent_pose = get_agent_pose(controller)
    desired_yaw = compute_yaw_to_target(
        agent_pose.get("position", {}),
        selected_candidate["position"],
    )
    for action in rotate_actions_to_yaw(
        float(agent_pose.get("rotation", {}).get("y", 0.0)),
        desired_yaw,
    ):
        if not execute_scan_action(action, should_check=False):
            break

    read_current_view("current")
    if not target_visible_within_success_distance:
        scan_actions = [
            ("LookDown", True),
            ("LookUp", False),
            ("RotateRight", True),
            ("RotateRight", True),
            ("RotateRight", True),
            ("RotateRight", False),
        ]
        for action, should_check in scan_actions:
            if not execute_scan_action(action, should_check=should_check):
                break
            if target_visible_within_success_distance:
                break

    planned_count = len(inspector._get_inspection_poses(selected_candidate))
    coverage_increment = 1.0 / planned_count if planned_count else 0.25
    local_viewpoint_id = (
        visited_viewpoint_id or f"{selected_instance_id}:local_scan:{decision_step}"
    )
    visited_viewpoint_ids = set(getattr(selected, "visited_viewpoint_ids", set()))
    visited_viewpoint_ids.add(local_viewpoint_id)
    coverage = min(1.0, max(selected.coverage, selected.coverage + coverage_increment))
    return (
        {
            "coverage": float(coverage),
            "evidence": 1.0 if evidence_found else 0.0,
            "evidence_visible": 1.0 if target_visible else 0.0,
            "evidence_support": (
                1.0
                if selected_instance_id == episode.get("true_support_instance_id")
                else 0.0
            ),
            "target_visible": target_visible,
            "target_distance": min_target_distance,
            "target_checks": target_checks,
            "target_visible_within_success_distance": (
                target_visible_within_success_distance
            ),
            "finish_inspection": True,
            "visited_viewpoint_id": local_viewpoint_id,
            "num_inspection_poses": planned_count,
            "visited_viewpoint_ids": visited_viewpoint_ids,
            "newly_visited_viewpoint_ids": {local_viewpoint_id},
            "total_inspection_poses": planned_count,
            "num_visited_viewpoints": len(visited_viewpoint_ids),
            "local_scan": True,
            "local_scan_success": target_visible_within_success_distance
            or (1.0 if evidence_found else 0.0) > tau_e,
            "success_trigger_action": success_trigger_action,
            "success_trigger_step": success_trigger_step,
        },
        global_step,
        representative_frame_path,
    )


def compute_switch_diagnostics(
    selected_sequence: list[str],
    wrong_instance_id: str | None,
    episode_type: str | None,
) -> tuple[bool, int | None]:
    if wrong_instance_id is None:
        return False, None

    first_wrong_index = None
    switched_from_wrong = False
    time_to_switch = None
    for index, instance_id in enumerate(selected_sequence):
        if instance_id == wrong_instance_id and first_wrong_index is None:
            first_wrong_index = index
        elif first_wrong_index is not None and instance_id != wrong_instance_id:
            switched_from_wrong = True
            time_to_switch = index - first_wrong_index
            break

    if episode_type != "misleading-prior":
        time_to_switch = None
    return switched_from_wrong, time_to_switch


def infer_switch_reason(
    method: str,
    previous_instance_id: str,
    selected_instance_id: str,
    pending_switch_reason: str | None,
    accessibility: dict[str, float],
    memory: SemanticInstanceMemory,
) -> str:
    if pending_switch_reason is not None:
        return pending_switch_reason
    if accessibility.get(previous_instance_id, 1.0) <= 0.0:
        return "accessibility_zero"
    if method == "sp_visit_penalty":
        return "visit_penalty"
    previous_instance = memory.get_instance(previous_instance_id)
    if previous_instance is not None and previous_instance.reliability < 1.0:
        return "reliability_decay"
    return "utility_change"


def compute_policy_switch_diagnostics(
    switch_events: list[dict[str, Any]],
    wrong_instance_id: str | None,
    episode_type: str | None,
) -> tuple[bool, bool, str | None]:
    if episode_type != "misleading-prior" or wrong_instance_id is None:
        return False, False, None
    for event in switch_events:
        if (
            event.get("switch_from") == wrong_instance_id
            and event.get("switch_to") != wrong_instance_id
        ):
            switch_reason = event.get("switch_reason")
            navigation_forced = bool(event.get("navigation_forced_switch", False))
            return (not navigation_forced), navigation_forced, switch_reason
    return False, False, None


def make_error_summary(
    episode: dict[str, Any],
    method: str,
    error: Exception,
) -> dict[str, Any]:
    return {
        "episode_id": episode.get("episode_id"),
        "method": METHOD_LABELS[method],
        "episode_type": episode.get("episode_type"),
        "target_category": episode.get("target_category"),
        "scene": episode.get("scene"),
        "success": False,
        "stopped": False,
        "terminal_action": None,
        "error": str(error),
        "num_actions": 0,
        "num_decisions": 0,
        "path_length": 0.0,
        "selected_sequence": [],
        "wrong_instance_id": episode.get("wrong_instance_id"),
        "true_support_instance_id": episode.get("true_support_instance_id"),
        "true_support_selected": False,
        "true_support_first_decision": None,
        "true_support_inspect_count": 0,
        "true_support_max_evidence": 0.0,
        "true_support_max_visible_evidence": 0.0,
        "true_support_support_evidence": 0.0,
        "true_support_target_visible_any": False,
        "target_seen_any": False,
        "oracle_support_evidence": False,
        "success_source": None,
        "success_selected_instance_id": None,
        "success_selected_instance_alias": None,
        "success_is_true_support": False,
        "visible_before_true_support": False,
        "target_seen_from_non_true_support": False,
        "spf_trigger_count": 0,
        "wrong_prior_decisions": 0,
        "switched_from_wrong": False,
        "time_to_switch": None,
        "switch_events": [],
        "switched_from_wrong_policy": False,
        "navigation_forced_switch": False,
        "first_switch_from_wrong_reason": None,
        "stop_precision": 0,
    }


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
    frame_saver: FrameSaver | None = None,
    oracle_support_evidence: bool = False,
) -> dict[str, Any]:
    memory = initialize_memory(episode, tau_c=tau_c, tau_e=tau_e, tau_n=tau_n)
    candidates_by_id = {
        candidate["instance_id"]: candidate
        for candidate in episode["candidate_instances"]
    }
    visit_counts = {instance_id: 0 for instance_id in candidates_by_id}
    accessibility = {instance_id: 1.0 for instance_id in candidates_by_id}
    planner = GridPlanner(controller)
    inspector = ControlledInspector(
        controller,
        reachable_positions=planner.reachable_positions,
    )
    wrong_id = episode.get("wrong_instance_id")
    true_id = episode.get("true_support_instance_id")

    global_step = 0
    path_length = 0.0
    selected_sequence: list[str] = []
    success = False
    stopped = False
    terminal_action = None
    active_instance_id = None
    spf_trigger_count = 0
    wrong_prior_decisions = 0
    partial_failure_error = None
    consecutive_navigation_failures = 0
    max_viewpoint_retries = 3
    scan_radius = 1.25
    success_distance = float(episode.get("success_distance", 1.5))
    true_support_selected = False
    true_support_first_decision = None
    true_support_max_evidence = 0.0
    true_support_max_visible_evidence = 0.0
    true_support_support_evidence = 0.0
    true_support_target_visible_any = False
    target_seen_any = False
    success_source = None
    success_selected_instance_id = None
    success_selected_instance_alias = None
    last_selected_instance_id: str | None = None
    pending_switch_reason: str | None = None
    switch_events: list[dict[str, Any]] = []

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
                accessibility,
            )
        if selected is None:
            break
        switch_reason = None
        navigation_forced_switch = False
        if (
            last_selected_instance_id is not None
            and selected.instance_id != last_selected_instance_id
        ):
            switch_reason = infer_switch_reason(
                method,
                last_selected_instance_id,
                selected.instance_id,
                pending_switch_reason,
                accessibility,
                memory,
            )
            navigation_forced_switch = (
                switch_reason in NAVIGATION_FORCED_SWITCH_REASONS
            )
            switch_events.append(
                {
                    "decision_step": decision_step,
                    "switch_from": last_selected_instance_id,
                    "switch_to": selected.instance_id,
                    "switch_reason": switch_reason,
                    "navigation_forced_switch": navigation_forced_switch,
                }
            )
        pending_switch_reason = None
        last_selected_instance_id = selected.instance_id
        selected_sequence.append(selected.instance_id)
        if selected.instance_id == true_id:
            true_support_selected = True
            if true_support_first_decision is None:
                true_support_first_decision = decision_step
        if selected.instance_id == wrong_id:
            wrong_prior_decisions += 1
        utility = compute_utility(
            memory,
            selected,
            method,
            visit_counts,
            beta_v,
            accessibility,
        )

        agent_pose = get_agent_pose(controller)
        selected_candidate = candidates_by_id[selected.instance_id]
        navigation_skipped_for_active_inspection = (
            selection_reason == "continue_active_inspection"
            and is_agent_in_inspection_region(
                agent_pose,
                inspector,
                selected_candidate,
                tolerance=planner.grid_size * 0.75,
            )
        )
        navigation_failed_for_instance = False
        local_scan_observation = None
        local_scan_representative_frame_path = None
        current_navigation_viewpoint_id = None
        active_release_reason = None
        if navigation_skipped_for_active_inspection:
            navigation_succeeded = True
        else:
            navigation_succeeded = False
            navigation_error = None
            failed_viewpoint_ids = getattr(selected, "failed_viewpoint_ids", set())
            pose_items = inspection_pose_items_for_navigation(
                inspector,
                selected_candidate,
                getattr(selected, "visited_viewpoint_ids", set()),
                failed_viewpoint_ids,
                max_viewpoint_retries,
            )
            for viewpoint_id, inspection_pose in pose_items:
                navigation_succeeded, global_step, path_length, navigation_error = (
                    navigate_to_pose(
                        controller,
                        planner,
                        logger,
                        episode,
                        method,
                        selected,
                        decision_step,
                        inspection_pose,
                        max_steps,
                        global_step,
                        path_length,
                        success,
                        utility,
                        selection_reason,
                        frame_saver,
                        navigation_skipped_for_active_inspection,
                    )
                )
                if navigation_succeeded or global_step >= max_steps:
                    if navigation_succeeded:
                        current_navigation_viewpoint_id = viewpoint_id
                    break
                if (
                    horizontal_distance_to_position(
                        get_agent_pose(controller),
                        selected_candidate["position"],
                    )
                    < scan_radius
                ):
                    (
                        local_scan_observation,
                        global_step,
                        local_scan_representative_frame_path,
                    ) = perform_local_scan(
                        controller,
                        logger,
                        episode,
                        method,
                        selected,
                        selected_candidate,
                        inspector,
                        decision_step,
                        global_step,
                        path_length,
                        success,
                        utility,
                        selection_reason,
                        frame_saver,
                        tau_e,
                        success_distance,
                        viewpoint_id,
                    )
                    navigation_succeeded = True
                    current_navigation_viewpoint_id = viewpoint_id
                    break
                failed_viewpoint_ids.add(viewpoint_id)
                selected.failed_viewpoint_ids = failed_viewpoint_ids

            if not navigation_succeeded:
                all_viewpoint_ids = all_inspection_viewpoint_ids(
                    inspector,
                    selected_candidate,
                )
                failed_viewpoint_ids = getattr(
                    selected, "failed_viewpoint_ids", set()
                )
                all_viewpoints_failed = (not all_viewpoint_ids) or (
                    all_viewpoint_ids <= failed_viewpoint_ids
                )
                if not all_viewpoints_failed:
                    selected.active_inspection = True
                    active_instance_id = selected.instance_id
                    continue

                navigation_failed_for_instance = True
                accessibility[selected.instance_id] = 0.0
                selected.accessibility = 0.0
                selected.active_inspection = False
                active_release_reason = "accessibility_release"
                pending_switch_reason = "navigation_failed"
                if active_instance_id == selected.instance_id:
                    active_instance_id = None
                consecutive_navigation_failures += 1
                log_navigation_failed_for_instance(
                    logger,
                    episode,
                    method,
                    global_step,
                    decision_step,
                    selected,
                    path_length,
                    active_release_reason,
                    navigation_error or "all_inspection_viewpoints_failed",
                    failed_viewpoint_ids,
                    active_release_reason,
                )
                if consecutive_navigation_failures > 5:
                    partial_failure_error = (
                        "low_level_action_failed: consecutive_navigation_failures"
                    )
                    success = False
                    break
                continue

        consecutive_navigation_failures = 0

        if partial_failure_error is not None or global_step >= max_steps:
            break

        reliability_before = selected.reliability
        if local_scan_observation is not None:
            observation = local_scan_observation
        else:
            (
                observation,
                global_step,
                local_scan_representative_frame_path,
            ) = perform_local_scan(
                controller,
                logger,
                episode,
                method,
                selected,
                selected_candidate,
                inspector,
                decision_step,
                global_step,
                path_length,
                success,
                utility,
                selection_reason,
                frame_saver,
                tau_e,
                success_distance,
                current_navigation_viewpoint_id,
            )
        if oracle_support_evidence and selected.instance_id == true_id:
            observation["evidence"] = 1.0
            observation["oracle_support_evidence"] = True
            observation["success_trigger_action"] = "OracleSupport"
            observation["success_trigger_step"] = global_step
        spf_triggered, reliability = update_memory_after_inspection(
            memory, selected, observation, method
        )
        if spf_triggered:
            spf_trigger_count += 1
        selected.reliability = reliability
        target_seen_any = target_seen_any or bool(
            observation.get("target_visible", False)
        )
        if selected.instance_id == true_id:
            true_support_max_evidence = max(
                true_support_max_evidence,
                float(observation.get("evidence", 0.0)),
            )
            true_support_max_visible_evidence = max(
                true_support_max_visible_evidence,
                float(observation.get("evidence_visible", 0.0)),
            )
            true_support_support_evidence = max(
                true_support_support_evidence,
                float(observation.get("evidence_support", 0.0)),
            )
            true_support_target_visible_any = (
                true_support_target_visible_any
                or bool(observation.get("target_visible", False))
            )
        if method == "ours" and partial_inspection:
            selected.active_inspection = True
            active_instance_id = selected.instance_id
            if selected.evidence > tau_e:
                active_release_reason = "success_release"
                selection_reason = "success_release"
                selected.active_inspection = False
                active_instance_id = None
            elif spf_triggered:
                active_release_reason = "spf_release"
                selection_reason = "spf_release"
                selected.active_inspection = False
                active_instance_id = None
                pending_switch_reason = "reliability_decay"
        if spf_triggered and pending_switch_reason is None:
            pending_switch_reason = "reliability_decay"
        if method == "sp_visit_penalty":
            visit_counts[selected.instance_id] += 1
            pending_switch_reason = "visit_penalty"

        global_step += 1
        success = (
            float(observation.get("evidence", 0.0)) > tau_e
            or bool(observation.get("target_visible_within_success_distance", False))
        )
        if success:
            success_selected_instance_id = selected.instance_id
            success_selected_instance_alias = selected.alias
            if bool(observation.get("oracle_support_evidence", False)):
                success_source = "oracle_support"
            else:
                success_source = "visible_target"
        success_info = (
            make_success_info(
                observation,
                selected,
                true_id,
                success_source,
                global_step,
            )
            if success
            else None
        )
        frame_path = (
            frame_saver.maybe_save(controller.last_event.frame, global_step)
            if frame_saver is not None
            else None
        )
        target_frame_info = (
            extract_target_frame_info(controller.last_event, episode)
            if frame_path is not None
            else None
        )
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
            frame_path=frame_path,
            phase="inspection_action",
            reliability_before=reliability_before,
            reliability_after=reliability,
            target_frame_info=target_frame_info,
            navigation_skipped_for_active_inspection=(
                navigation_skipped_for_active_inspection
            ),
            active_release_reason=active_release_reason,
            observation_info=observation,
            success_info=success_info,
            switch_reason=switch_reason,
            navigation_forced_switch=navigation_forced_switch,
        )
        log_decision_update(
            logger,
            episode,
            method,
            global_step,
            decision_step,
            selected,
            path_length,
            success,
            utility,
            spf_triggered,
            selection_reason,
            reliability_before,
            reliability,
            frame_path or local_scan_representative_frame_path,
            navigation_skipped_for_active_inspection,
            active_release_reason,
            success_info,
            switch_reason,
            navigation_forced_switch,
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
                phase="virtual_stop",
                reliability_before=reliability_before,
                reliability_after=reliability,
                navigation_skipped_for_active_inspection=(
                    navigation_skipped_for_active_inspection
                ),
                active_release_reason=active_release_reason,
                observation_info=observation,
                success_info=success_info,
                switch_reason=switch_reason,
                navigation_forced_switch=navigation_forced_switch,
            )
            stopped = True
            terminal_action = "Stop"
            break

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
    true_support_inspect_count = (
        memory.get_instance(true_id).inspect_count
        if true_id in candidates_by_id
        else 0
    )
    switched_from_wrong, time_to_switch = compute_switch_diagnostics(
        selected_sequence,
        wrong_id,
        episode.get("episode_type"),
    )
    (
        switched_from_wrong_policy,
        navigation_forced_switch,
        first_switch_from_wrong_reason,
    ) = compute_policy_switch_diagnostics(
        switch_events,
        wrong_id,
        episode.get("episode_type"),
    )
    success_is_true_support = success_selected_instance_id == true_id
    summary = {
        "episode_id": episode["episode_id"],
        "method": METHOD_LABELS[method],
        "episode_type": episode.get("episode_type"),
        "target_category": episode.get("target_category"),
        "scene": episode.get("scene"),
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
        "true_support_selected": true_support_selected,
        "true_support_first_decision": true_support_first_decision,
        "true_support_inspect_count": true_support_inspect_count,
        "true_support_max_evidence": true_support_max_evidence,
        "true_support_max_visible_evidence": true_support_max_visible_evidence,
        "true_support_support_evidence": true_support_support_evidence,
        "true_support_target_visible_any": true_support_target_visible_any,
        "target_seen_any": target_seen_any,
        "oracle_support_evidence": oracle_support_evidence,
        "success_source": success_source,
        "success_selected_instance_id": success_selected_instance_id,
        "success_selected_instance_alias": success_selected_instance_alias,
        "success_is_true_support": success_is_true_support,
        "visible_before_true_support": (
            success and target_seen_any and not true_support_selected
        ),
        "target_seen_from_non_true_support": (
            success and target_seen_any and not success_is_true_support
        ),
        "reliability_drop_wrong": (
            None if final_wrong is None else 1.0 - final_wrong
        ),
        "spf_trigger_count": spf_trigger_count,
        "wrong_prior_decisions": wrong_prior_decisions,
        "time_to_switch": time_to_switch,
        "switched_from_wrong": switched_from_wrong,
        "switch_events": switch_events,
        "switched_from_wrong_policy": switched_from_wrong_policy,
        "navigation_forced_switch": navigation_forced_switch,
        "first_switch_from_wrong_reason": first_switch_from_wrong_reason,
        "stop_precision": 1 if stopped and success else 0,
    }
    if partial_failure_error is not None:
        summary.update(
            {
                "success": False,
                "error": partial_failure_error,
                "partial_failure": True,
                "num_actions": global_step,
                "num_decisions": len(selected_sequence),
                "selected_sequence": selected_sequence,
                "stopped": False,
                "terminal_action": None,
                "stop_precision": 0,
            }
        )
    return summary


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
    parser.add_argument("--save-rgb-frames", action="store_true")
    parser.add_argument("--rgb-output-dir", type=str, default=None)
    parser.add_argument("--save-frame-every", type=int, default=1)
    parser.add_argument("--frame-width", type=int, default=640)
    parser.add_argument("--frame-height", type=int, default=480)
    parser.add_argument("--oracle-support-evidence", action="store_true")
    args = parser.parse_args()

    episode = load_episode(args.episodes, args.episode_index)
    logger = EpisodeLogger(
        args.output.parent, args.output.name, METHOD_LABELS[args.method]
    )
    frame_saver = None
    if args.save_rgb_frames:
        rgb_output_dir = (
            Path(args.rgb_output_dir)
            if args.rgb_output_dir is not None
            else logger.episode_dir / "rgb_frames"
        )
        frame_saver = FrameSaver(rgb_output_dir, args.save_frame_every)
    controller = None
    try:
        from ai2thor.controller import Controller

        controller = Controller(
            scene=episode["scene"],
            width=args.frame_width if args.save_rgb_frames else 300,
            height=args.frame_height if args.save_rgb_frames else 300,
            gridSize=0.25,
            renderDepthImage=False,
            renderInstanceSegmentation=args.save_rgb_frames,
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
            frame_saver=frame_saver,
            oracle_support_evidence=args.oracle_support_evidence,
        )
        logger.save_summary(summary)
        print(f"success={summary['success']}")
        print(f"num_actions={summary['num_actions']}")
        print(f"path_length={summary['path_length']:.3f}")
        print(f"step log: {logger.step_log_path}")
        print(f"episode summary: {logger.summary_path}")
    except Exception as error:
        summary = make_error_summary(episode, args.method, error)
        logger.save_summary(summary)
        print(f"success={summary['success']}")
        print(f"error={summary['error']}")
        print(f"episode summary: {logger.summary_path}")
    finally:
        if controller is not None:
            controller.stop()
        logger.close()


if __name__ == "__main__":
    main()

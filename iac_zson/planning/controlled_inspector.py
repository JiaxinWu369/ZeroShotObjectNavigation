"""Single-instance controlled inspection using AI2-THOR teleport actions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from iac_zson.planning.inspection_points import get_nearest_inspection_poses


class ControlledInspector:
    """Inspect one candidate from nearby reachable viewpoints."""

    def __init__(
        self,
        controller: Any,
        k: int = 4,
        rotations_per_pose: int = 4,
    ) -> None:
        self.controller = controller
        self.k = k
        self.rotations_per_pose = rotations_per_pose

    def inspect(
        self,
        episode: Mapping[str, Any],
        selected_instance: Mapping[str, Any],
        poses_per_decision: Optional[int] = None,
        visited_viewpoint_ids: Optional[set] = None,
    ) -> dict:
        reachable_event = self.controller.step(action="GetReachablePositions")
        reachable_positions = reachable_event.metadata.get("actionReturn") or []
        inspection_poses = get_nearest_inspection_poses(
            selected_instance["position"], reachable_positions, self.k
        )

        target_category = episode["target_category"]
        selected_instance_id = selected_instance["instance_id"]
        target_visible = False
        evidence_found = False
        known_visited_viewpoint_ids = set(visited_viewpoint_ids or set())
        newly_visited_viewpoint_ids = set()
        last_viewpoint_id = ""

        pose_items = list(enumerate(inspection_poses))
        if poses_per_decision is not None:
            pose_items = [
                (pose_index, pose)
                for pose_index, pose in pose_items
                if self._viewpoint_id(selected_instance_id, pose_index)
                not in known_visited_viewpoint_ids
            ][:poses_per_decision]

        for pose_index, pose in pose_items:
            self.controller.step(
                action="TeleportFull",
                x=pose["x"],
                y=pose["y"],
                z=pose["z"],
                rotation={"x": 0.0, "y": pose["rotation_y"], "z": 0.0},
                horizon=pose["horizon"],
                standing=True,
            )
            metadata = self.controller.last_event.metadata
            if not metadata.get("lastActionSuccess", True):
                continue

            last_viewpoint_id = self._viewpoint_id(
                selected_instance_id, pose_index
            )
            known_visited_viewpoint_ids.add(last_viewpoint_id)
            newly_visited_viewpoint_ids.add(last_viewpoint_id)
            visible_now, evidence_now = self._read_observation(
                metadata, target_category, selected_instance_id
            )
            target_visible = target_visible or visible_now
            evidence_found = evidence_found or evidence_now

            for _ in range(self.rotations_per_pose):
                self.controller.step(action="RotateRight")
                visible_now, evidence_now = self._read_observation(
                    self.controller.last_event.metadata,
                    target_category,
                    selected_instance_id,
                )
                target_visible = target_visible or visible_now
                evidence_found = evidence_found or evidence_now

        planned_count = len(inspection_poses)
        visited_count = len(known_visited_viewpoint_ids)
        coverage = visited_count / planned_count if planned_count else 0.0
        return {
            "coverage": float(coverage),
            "evidence": 1.0 if evidence_found else 0.0,
            "target_visible": target_visible,
            "finish_inspection": True,
            "visited_viewpoint_id": last_viewpoint_id,
            "num_inspection_poses": planned_count,
            "visited_viewpoint_ids": known_visited_viewpoint_ids,
            "newly_visited_viewpoint_ids": newly_visited_viewpoint_ids,
            "total_inspection_poses": planned_count,
            "num_visited_viewpoints": visited_count,
        }

    @staticmethod
    def _viewpoint_id(selected_instance_id: str, pose_index: int) -> str:
        return f"{selected_instance_id}:inspection:{pose_index}"

    @staticmethod
    def _read_observation(
        metadata: Mapping[str, Any],
        target_category: str,
        selected_instance_id: str,
    ) -> tuple[bool, bool]:
        target_visible = False
        evidence = False
        for obj in metadata.get("objects", []) or []:
            if (
                not isinstance(obj, Mapping)
                or obj.get("objectType") != target_category
                or not obj.get("visible", False)
            ):
                continue
            target_visible = True
            parents = obj.get("parentReceptacles") or []
            if selected_instance_id in parents:
                evidence = True
        return target_visible, evidence


def run_controlled_inspection(
    controller: Any,
    episode: Mapping[str, Any],
    selected_instance: Mapping[str, Any],
    k: int = 4,
    rotations_per_pose: int = 4,
    poses_per_decision: Optional[int] = None,
    visited_viewpoint_ids: Optional[set] = None,
) -> dict:
    """Convenience wrapper for one controlled inspection."""
    return ControlledInspector(controller, k, rotations_per_pose).inspect(
        episode,
        selected_instance,
        poses_per_decision=poses_per_decision,
        visited_viewpoint_ids=visited_viewpoint_ids,
    )

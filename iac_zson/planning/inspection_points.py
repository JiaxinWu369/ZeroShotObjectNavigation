"""Geometry helpers for controlled instance inspection."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def compute_yaw_to_target(
    agent_pos: Mapping[str, float], target_pos: Mapping[str, float]
) -> float:
    """Return a yaw angle in degrees that faces the target in the x-z plane."""
    delta_x = float(target_pos["x"]) - float(agent_pos["x"])
    delta_z = float(target_pos["z"]) - float(agent_pos["z"])
    if delta_x == 0.0 and delta_z == 0.0:
        return 0.0
    return math.degrees(math.atan2(delta_x, delta_z)) % 360.0


def get_nearest_inspection_poses(
    instance_position: Mapping[str, float],
    reachable_positions: Sequence[Mapping[str, Any]],
    k: int = 4,
) -> list[dict[str, float]]:
    """Return the nearest reachable poses, oriented toward an instance."""
    if not reachable_positions or k <= 0:
        return []

    target_x = float(instance_position["x"])
    target_y = float(instance_position.get("y", 0.0))
    target_z = float(instance_position["z"])

    ranked = []
    for index, position in enumerate(reachable_positions):
        try:
            x = float(position["x"])
            y = float(position.get("y", 0.0))
            z = float(position["z"])
        except (KeyError, TypeError, ValueError):
            continue
        distance_squared = (
            (x - target_x) ** 2
            + (y - target_y) ** 2
            + (z - target_z) ** 2
        )
        ranked.append((distance_squared, index, x, y, z))

    ranked.sort(key=lambda item: (item[0], item[1]))
    target = {"x": target_x, "y": target_y, "z": target_z}
    poses = []
    for _, _, x, y, z in ranked[:k]:
        position = {"x": x, "y": y, "z": z}
        poses.append(
            {
                "x": x,
                "y": y,
                "z": z,
                "rotation_y": compute_yaw_to_target(position, target),
                "horizon": 0.0,
            }
        )
    return poses


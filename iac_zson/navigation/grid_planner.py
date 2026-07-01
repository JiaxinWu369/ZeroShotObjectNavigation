"""Grid planner over AI2-THOR reachable positions."""

from __future__ import annotations

from collections import deque
import heapq
import math
from typing import Any


class GridPlanner:
    """Build a simple 4-neighbor grid graph from reachable positions."""

    def __init__(
        self,
        controller: Any,
        grid_size: float = 0.25,
        debug: bool = False,
    ) -> None:
        self.controller = controller
        self.grid_size = grid_size
        self.debug = debug
        self.reachable_positions = self._get_reachable_positions()
        self.nodes = {
            self._to_key(position): position
            for position in self.reachable_positions
        }

    def _get_reachable_positions(self) -> list[dict[str, float]]:
        event = self.controller.step(action="GetReachablePositions")
        return event.metadata.get("actionReturn") or []

    def _to_key(self, position: dict[str, float]) -> tuple[int, int]:
        return (
            round(float(position["x"]) / self.grid_size),
            round(float(position["z"]) / self.grid_size),
        )

    def _from_key(self, key: tuple[int, int]) -> dict[str, float]:
        position = self.nodes[key]
        return {
            "x": float(position["x"]),
            "y": float(position.get("y", 0.0)),
            "z": float(position["z"]),
        }

    @staticmethod
    def _distance_xz(a: dict[str, float], b: dict[str, float]) -> float:
        dx = float(a["x"]) - float(b["x"])
        dz = float(a["z"]) - float(b["z"])
        return math.sqrt(dx * dx + dz * dz)

    def nearest_reachable(self, position: dict[str, float]) -> dict[str, float] | None:
        """Return the reachable point nearest to ``position`` in the x-z plane."""
        if not self.reachable_positions:
            return None
        nearest = min(
            self.reachable_positions,
            key=lambda candidate: self._distance_xz(candidate, position),
        )
        return {
            "x": float(nearest["x"]),
            "y": float(nearest.get("y", 0.0)),
            "z": float(nearest["z"]),
        }

    def _neighbors(self, key: tuple[int, int]) -> list[tuple[int, int]]:
        x, z = key
        candidates = [
            (x + 1, z),
            (x - 1, z),
            (x, z + 1),
            (x, z - 1),
        ]
        return [candidate for candidate in candidates if candidate in self.nodes]

    @staticmethod
    def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _debug(self, message: str) -> None:
        if self.debug:
            print(message)

    def _is_valid_path(self, path: list[dict[str, float]]) -> bool:
        tolerance = max(1e-4, self.grid_size * 0.2)
        for current, next_position in zip(path, path[1:]):
            distance = self._distance_xz(current, next_position)
            if abs(distance - self.grid_size) > tolerance:
                self._debug(
                    "invalid path edge: "
                    f"current position={current}, "
                    f"next position={next_position}, "
                    f"distance={distance:.4f}, "
                    f"grid_size={self.grid_size:.4f}"
                )
                return False
        return True

    def shortest_path(
        self,
        start_position: dict[str, float],
        goal_position: dict[str, float],
    ) -> list[dict[str, float]]:
        """Return a shortest reachable grid path from start to goal.

        The returned path includes both start and goal reachable points. If no
        path is found, an empty list is returned.
        """
        start = self.nearest_reachable(start_position)
        goal = self.nearest_reachable(goal_position)
        if start is None or goal is None:
            return []

        start_key = self._to_key(start)
        goal_key = self._to_key(goal)
        if start_key == goal_key:
            return [self._from_key(start_key)]

        frontier: list[tuple[int, tuple[int, int]]] = []
        heapq.heappush(frontier, (0, start_key))
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {
            start_key: None
        }
        cost_so_far = {start_key: 0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal_key:
                break

            for neighbor in self._neighbors(current):
                new_cost = cost_so_far[current] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self._heuristic(neighbor, goal_key)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current

        if goal_key not in came_from:
            return []

        path_keys = []
        current: tuple[int, int] | None = goal_key
        while current is not None:
            path_keys.append(current)
            current = came_from[current]
        path_keys.reverse()
        path = [self._from_key(key) for key in path_keys]
        if not self._is_valid_path(path):
            return []
        return path

    @staticmethod
    def _normalize_yaw(rotation_y: float) -> int:
        return int(round(rotation_y / 90.0) * 90) % 360

    @staticmethod
    def _yaw_to_step(start: dict[str, float], end: dict[str, float]) -> int:
        dx = round(float(end["x"]) - float(start["x"]), 6)
        dz = round(float(end["z"]) - float(start["z"]), 6)
        if abs(dx) >= abs(dz):
            return 90 if dx > 0 else 270
        return 0 if dz > 0 else 180

    @staticmethod
    def _rotation_actions(current_yaw: int, target_yaw: int) -> tuple[list[str], int]:
        diff = (target_yaw - current_yaw) % 360
        if diff == 0:
            return [], current_yaw
        if diff == 90:
            return ["RotateRight"], target_yaw
        if diff == 180:
            return ["RotateRight", "RotateRight"], target_yaw
        if diff == 270:
            return ["RotateLeft"], target_yaw
        return [], current_yaw

    def path_to_actions(
        self,
        start_pose: dict[str, Any],
        path: list[dict[str, float]],
    ) -> list[str]:
        """Convert a grid path into RotateLeft/RotateRight/MoveAhead actions."""
        if len(path) < 2:
            return []

        rotation = start_pose.get("rotation", {})
        current_yaw = self._normalize_yaw(float(rotation.get("y", 0.0)))
        actions: list[str] = []

        for start, end in zip(path, path[1:]):
            target_yaw = self._yaw_to_step(start, end)
            rotate_actions, current_yaw = self._rotation_actions(
                current_yaw, target_yaw
            )
            step_actions = list(rotate_actions)
            if current_yaw != target_yaw:
                self._debug(
                    "invalid yaw before MoveAhead: "
                    f"current position={start}, "
                    f"next position={end}, "
                    f"desired yaw={target_yaw}, "
                    f"current yaw={current_yaw}, "
                    f"generated actions={step_actions}"
                )
                return []
            actions.extend(rotate_actions)
            actions.append("MoveAhead")
            step_actions.append("MoveAhead")
            self._debug(
                "path action step: "
                f"current position={start}, "
                f"next position={end}, "
                f"desired yaw={target_yaw}, "
                f"current yaw={current_yaw}, "
                f"generated actions={step_actions}"
            )
        return actions


def bfs_shortest_path(
    nodes: set[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    """Small standalone BFS helper for tests or debugging."""
    queue = deque([start])
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while queue:
        current = queue.popleft()
        if current == goal:
            break
        x, z = current
        for neighbor in ((x + 1, z), (x - 1, z), (x, z + 1), (x, z - 1)):
            if neighbor in nodes and neighbor not in came_from:
                came_from[neighbor] = current
                queue.append(neighbor)
    if goal not in came_from:
        return []
    path = []
    current: tuple[int, int] | None = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    return list(reversed(path))

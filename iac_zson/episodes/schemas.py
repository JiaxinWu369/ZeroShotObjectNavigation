"""Schemas for diagnostic object-navigation episodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass
class DiagnosticEpisode:
    """Serializable metadata for one diagnostic episode."""

    episode_id: str
    scene: str
    scene_type: str
    target_category: str
    episode_type: str
    start_pose: dict[str, Any]
    candidate_instances: list[dict[str, Any]]
    wrong_instance_id: Optional[str]
    true_support_instance_id: str
    target_object_id: str
    max_steps: int
    success_distance: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


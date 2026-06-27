"""Build diagnostic episodes from AI2-THOR scene metadata."""

from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from iac_zson.env.scene_parser import (
    get_objects_by_type,
    get_receptacle_instances,
)
from iac_zson.episodes.schemas import DiagnosticEpisode


TARGET_CATEGORIES = ("Mug", "Bowl", "Book", "RemoteControl", "Towel")

DEFAULT_ALLOWED_CATEGORIES = frozenset(
    {
        "CounterTop",
        "DiningTable",
        "CoffeeTable",
        "Sink",
        "Cabinet",
        "Sofa",
        "Desk",
        "Shelf",
        "TVStand",
        "Bed",
        "Bathtub",
        "TowelHolder",
        "Dresser",
        "SideTable",
        "Drawer",
        "Fridge",
    }
)

OPTIONAL_ALLOWED_CATEGORIES = frozenset({"StoveBurner", "Microwave"})

EXCLUDED_SMALL_OBJECT_CATEGORIES = frozenset(
    {
        "Mug",
        "Bowl",
        "Book",
        "Cup",
        "Apple",
        "Bottle",
        "Plate",
        "Pot",
        "Pan",
        "Bread",
        "Toaster",
    }
)


def infer_scene_type(scene: str) -> str:
    """Infer the standard iTHOR room type from a FloorPlan number."""
    match = re.search(r"FloorPlan(\d+)", scene)
    if match is None:
        return "unknown"
    number = int(match.group(1))
    if 1 <= number <= 30:
        return "kitchen"
    if 31 <= number <= 60:
        return "living_room"
    if 61 <= number <= 90:
        return "bedroom"
    if 91 <= number <= 120:
        return "bathroom"
    return "unknown"


class DiagnosticEpisodeBuilder:
    """Create normal- and misleading-prior episodes without agent actions."""

    def __init__(
        self,
        controller: Any,
        priors: Mapping[str, Mapping[str, float] | Sequence[str]],
        seed: int = 42,
        max_steps: int = 500,
        success_distance: float = 1.0,
        allowed_categories: Iterable[str] | None = None,
        include_optional_categories: bool = False,
        target_categories: Sequence[str] = TARGET_CATEGORIES,
        min_true_prior: float = 0.3,
    ) -> None:
        self.controller = controller
        self.priors = {
            target: self._normalize_prior(value) for target, value in priors.items()
        }
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        self.success_distance = success_distance
        self.allowed_categories = set(
            DEFAULT_ALLOWED_CATEGORIES
            if allowed_categories is None
            else allowed_categories
        )
        if include_optional_categories:
            self.allowed_categories.update(OPTIONAL_ALLOWED_CATEGORIES)
        self.target_categories = tuple(target_categories)
        self.min_true_prior = min_true_prior

    def build(self, scenes: Sequence[str]) -> list[DiagnosticEpisode]:
        episodes = []
        for scene in scenes:
            episodes.extend(self.build_scene(scene))
        return episodes

    def build_scene(self, scene: str) -> list[DiagnosticEpisode]:
        reset_event = self.controller.reset(scene=scene)
        metadata = reset_event.metadata
        reachable_event = self.controller.step(action="GetReachablePositions")
        reachable_positions = reachable_event.metadata.get("actionReturn") or []
        if not reachable_positions:
            raise RuntimeError(f"No reachable positions found in scene {scene}")

        supports = get_receptacle_instances(metadata)
        supports_by_id = {
            support["object_id"]: support
            for support in supports
            if support["object_id"] is not None
        }
        episodes = []

        for target_category in self.target_categories:
            prior_scores = self.priors.get(target_category, {})
            for target_index, target in enumerate(
                get_objects_by_type(metadata, target_category)
            ):
                target_id = target["object_id"]
                if target_id is None:
                    continue

                parent_ids = [
                    parent_id
                    for parent_id in target["parent_receptacles"]
                    if parent_id in supports_by_id
                ]
                if not parent_ids:
                    continue

                true_support_id = self._best_support(
                    parent_ids, supports_by_id, prior_scores
                )
                true_category = supports_by_id[true_support_id]["object_type"]
                if true_category not in self.allowed_categories:
                    print(
                        f"SKIP scene={scene} target={target_id} "
                        f"reason=true_support_category_not_allowed "
                        f"category={true_category}"
                    )
                    continue

                candidate_supports = [
                    support
                    for support in supports
                    if self._is_candidate(support, target_category)
                ]
                candidates = self._candidate_instances(
                    candidate_supports, target_category
                )
                candidate_ids = {
                    candidate["instance_id"] for candidate in candidates
                }
                if true_support_id not in candidate_ids:
                    print(
                        f"SKIP scene={scene} target={target_id} "
                        f"reason=true_support_filtered "
                        f"true_support_instance_id={true_support_id}"
                    )
                    continue

                candidate_by_id = {
                    candidate["instance_id"]: candidate
                    for candidate in candidates
                }
                candidate_rank_by_id = {
                    candidate["instance_id"]: rank
                    for rank, candidate in enumerate(candidates, start=1)
                }
                true_support_p_sem = candidate_by_id[true_support_id][
                    "p_sem"
                ]
                true_support_rank = candidate_rank_by_id[true_support_id]
                episode_prefix = f"{scene}_{target_category}_{target_index:03d}"

                if true_category in prior_scores:
                    if true_support_p_sem < self.min_true_prior:
                        print(
                            f"SKIP scene={scene} target={target_id} "
                            "skip normal: true support prior too low"
                        )
                    else:
                        episodes.append(
                            self._make_episode(
                                episode_id=f"{episode_prefix}_normal_prior",
                                scene=scene,
                                target_category=target_category,
                                episode_type="normal-prior",
                                reachable_positions=reachable_positions,
                                candidates=candidates,
                                wrong_instance_id=None,
                                true_support_instance_id=true_support_id,
                                target_object_id=target_id,
                                true_support_p_sem=true_support_p_sem,
                                wrong_instance_p_sem=None,
                                true_support_rank=true_support_rank,
                                wrong_instance_rank=None,
                            )
                        )

                if true_support_p_sem < self.min_true_prior:
                    print(
                        f"SKIP scene={scene} target={target_id} "
                        "skip misleading: true support prior too low"
                    )
                    continue

                wrong_candidate = self._find_wrong_candidate(
                    candidates=candidates,
                    parent_ids=set(parent_ids),
                    target_object_id=target_id,
                    true_support_instance_id=true_support_id,
                    true_support_rank=true_support_rank,
                )
                if wrong_candidate is None:
                    print(
                        f"SKIP scene={scene} target={target_id} "
                        "skip misleading: no wrong instance ranked before true support"
                    )
                else:
                    wrong_id = wrong_candidate["instance_id"]
                    episodes.append(
                        self._make_episode(
                            episode_id=f"{episode_prefix}_misleading_prior",
                            scene=scene,
                            target_category=target_category,
                            episode_type="misleading-prior",
                            reachable_positions=reachable_positions,
                            candidates=candidates,
                            wrong_instance_id=wrong_id,
                            true_support_instance_id=true_support_id,
                            target_object_id=target_id,
                            true_support_p_sem=true_support_p_sem,
                            wrong_instance_p_sem=wrong_candidate["p_sem"],
                            true_support_rank=true_support_rank,
                            wrong_instance_rank=candidate_rank_by_id[
                                wrong_id
                            ],
                        )
                    )

        return episodes

    @staticmethod
    def _normalize_prior(
        prior: Mapping[str, float] | Sequence[str],
    ) -> dict[str, float]:
        if isinstance(prior, Mapping):
            return {category: float(score) for category, score in prior.items()}
        return {
            category: max(0.1, 1.0 - 0.1 * rank)
            for rank, category in enumerate(prior)
        }

    def _is_candidate(self, support: dict, target_category: str) -> bool:
        category = support.get("object_type")
        return (
            support.get("object_id") is not None
            and category in self.allowed_categories
            and category != "Floor"
            and category != target_category
            and category not in EXCLUDED_SMALL_OBJECT_CATEGORIES
            and not support.get("pickupable", False)
        )

    def _candidate_instances(
        self, supports: list[dict], target_category: str
    ) -> list[dict]:
        counts: defaultdict[str, int] = defaultdict(int)
        candidates = []
        prior_scores = self.priors.get(target_category, {})
        for support in supports:
            instance_id = support["object_id"]
            category = support["object_type"]
            if instance_id is None or category is None:
                continue
            counts[category] += 1
            candidates.append(
                {
                    "instance_id": instance_id,
                    "alias": f"{category}_{counts[category]}",
                    "category": category,
                    "position": support["position"],
                    "p_sem": prior_scores.get(category, 0.1),
                }
            )
        candidates.sort(
            key=lambda candidate: (
                -candidate["p_sem"],
                candidate["category"],
                candidate["alias"],
            )
        )
        return candidates

    def _best_support(
        self,
        support_ids: list[str],
        supports_by_id: Mapping[str, dict],
        prior_scores: Mapping[str, float],
    ) -> str:
        return min(
            support_ids,
            key=lambda instance_id: (
                -prior_scores.get(
                    supports_by_id[instance_id]["object_type"], 0.1
                ),
                instance_id,
            ),
        )

    @staticmethod
    def _find_wrong_candidate(
        candidates: list[dict],
        parent_ids: set[str],
        target_object_id: str,
        true_support_instance_id: str,
        true_support_rank: int,
    ) -> dict | None:
        for rank, candidate in enumerate(candidates, start=1):
            instance_id = candidate["instance_id"]
            if (
                rank < true_support_rank
                and instance_id != true_support_instance_id
                and instance_id != target_object_id
                and instance_id not in parent_ids
            ):
                return candidate
        return None

    def _make_episode(
        self,
        episode_id: str,
        scene: str,
        target_category: str,
        episode_type: str,
        reachable_positions: list[dict],
        candidates: list[dict],
        wrong_instance_id: str | None,
        true_support_instance_id: str,
        target_object_id: str,
        true_support_p_sem: float | None = None,
        wrong_instance_p_sem: float | None = None,
        true_support_rank: int | None = None,
        wrong_instance_rank: int | None = None,
    ) -> DiagnosticEpisode:
        position = dict(self.rng.choice(reachable_positions))
        start_pose = {
            "position": position,
            "rotation": {
                "x": 0.0,
                "y": float(self.rng.choice((0, 90, 180, 270))),
                "z": 0.0,
            },
            "horizon": 0.0,
            "standing": True,
        }
        return DiagnosticEpisode(
            episode_id=episode_id,
            scene=scene,
            scene_type=infer_scene_type(scene),
            target_category=target_category,
            episode_type=episode_type,
            start_pose=start_pose,
            candidate_instances=candidates,
            wrong_instance_id=wrong_instance_id,
            true_support_instance_id=true_support_instance_id,
            target_object_id=target_object_id,
            max_steps=self.max_steps,
            success_distance=self.success_distance,
            true_support_p_sem=true_support_p_sem,
            wrong_instance_p_sem=wrong_instance_p_sem,
            true_support_rank=true_support_rank,
            wrong_instance_rank=wrong_instance_rank,
        )

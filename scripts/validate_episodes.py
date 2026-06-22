"""Validate diagnostic episode JSONL files and print a compact report."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import json
import math
from numbers import Real
from pathlib import Path
from statistics import fmean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FIELDS = (
    "episode_id",
    "scene",
    "scene_type",
    "target_category",
    "episode_type",
    "start_pose",
    "candidate_instances",
    "true_support_instance_id",
    "target_object_id",
    "max_steps",
    "success_distance",
)

CANDIDATE_REQUIRED_FIELDS = ("instance_id", "alias", "category", "position")


def _candidate_ids(candidates: list[Any]) -> set[str]:
    return {
        candidate["instance_id"]
        for candidate in candidates
        if isinstance(candidate, Mapping)
        and isinstance(candidate.get("instance_id"), str)
    }


def _numeric_p_sem(candidate: Mapping[str, Any]) -> float | None:
    value = candidate.get("p_sem")
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    score = float(value)
    return score if math.isfinite(score) else None


def _candidate_by_id(candidates: list[Any], instance_id: Any) -> Mapping | None:
    for candidate in candidates:
        if (
            isinstance(candidate, Mapping)
            and candidate.get("instance_id") == instance_id
        ):
            return candidate
    return None


def print_top_candidates(episode: Mapping[str, Any], episode_id: str) -> None:
    """Print up to five candidates with the highest numeric p_sem."""
    candidates = episode.get("candidate_instances")
    ranked = []
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            score = _numeric_p_sem(candidate)
            if score is not None:
                ranked.append((score, candidate))

    ranked.sort(
        key=lambda item: (
            -item[0],
            str(item[1].get("alias", "")),
            str(item[1].get("instance_id", "")),
        )
    )
    true_support_id = episode.get("true_support_instance_id")
    wrong_id = episode.get("wrong_instance_id")
    print(f"top-5 candidates {episode_id}:")
    if not ranked:
        print("  (no candidates with numeric p_sem)")
        return

    for score, candidate in ranked[:5]:
        instance_id = candidate.get("instance_id")
        labels = []
        if instance_id == true_support_id:
            labels.append("TRUE")
        if instance_id == wrong_id:
            labels.append("WRONG")
        label = f" [{' / '.join(labels)}]" if labels else ""
        print(
            f"  {score:.4f}, {candidate.get('alias')}, "
            f"{candidate.get('category')}, {instance_id}{label}"
        )


def validate_episode(episode: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return validation errors and warnings for one episode."""
    errors = []
    warnings = []

    for field in REQUIRED_FIELDS:
        if field not in episode:
            errors.append(f"missing required field: {field}")

    candidates_value = episode.get("candidate_instances")
    if not isinstance(candidates_value, list):
        errors.append("candidate_instances must be a list")
        candidates = []
    else:
        candidates = candidates_value

    target_category = episode.get("target_category")
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            errors.append(f"candidate_instances[{index}] must be an object")
            continue

        for field in CANDIDATE_REQUIRED_FIELDS:
            if field not in candidate:
                errors.append(
                    f"candidate_instances[{index}] missing field: {field}"
                )

        instance_id = candidate.get("instance_id", f"index {index}")
        if "p_sem" not in candidate:
            warnings.append(f"candidate {instance_id} is missing p_sem")
        elif _numeric_p_sem(candidate) is None:
            warnings.append(f"candidate {instance_id} has non-numeric p_sem")

        category = candidate.get("category")
        if category == "Floor":
            warnings.append(f"candidate {instance_id} has Floor category")
        if category is not None and category == target_category:
            warnings.append(
                f"candidate {instance_id} has target object category {category}"
            )
        if candidate.get("pickupable", False):
            warnings.append(
                f"candidate {instance_id} has pickupable category {category}"
            )

    candidate_ids = _candidate_ids(candidates)
    episode_type = episode.get("episode_type")
    true_support_id = episode.get("true_support_instance_id")
    wrong_id = episode.get("wrong_instance_id")
    true_candidate = _candidate_by_id(candidates, true_support_id)
    true_p_sem = (
        _numeric_p_sem(true_candidate) if true_candidate is not None else None
    )
    if true_p_sem is not None and true_p_sem < 0.3:
        warnings.append(
            f"true support {true_support_id} has low p_sem {true_p_sem:.4f}"
        )

    if episode_type == "misleading-prior":
        if not wrong_id:
            errors.append("misleading-prior requires wrong_instance_id")
        elif wrong_id == true_support_id:
            errors.append(
                "wrong_instance_id must differ from true_support_instance_id"
            )
        if wrong_id and wrong_id not in candidate_ids:
            errors.append("wrong_instance_id is not in candidate_instances")
        if true_support_id not in candidate_ids:
            errors.append(
                "true_support_instance_id is not in candidate_instances"
            )
        wrong_candidate = _candidate_by_id(candidates, wrong_id)
        wrong_p_sem = (
            _numeric_p_sem(wrong_candidate)
            if wrong_candidate is not None
            else None
        )
        if (
            wrong_p_sem is not None
            and true_p_sem is not None
            and wrong_p_sem < true_p_sem
        ):
            warnings.append(
                f"wrong instance {wrong_id} p_sem {wrong_p_sem:.4f} is lower "
                f"than true support p_sem {true_p_sem:.4f}"
            )
    elif episode_type == "normal-prior":
        if wrong_id is not None:
            errors.append("normal-prior requires wrong_instance_id to be null")
        if true_support_id not in candidate_ids:
            errors.append(
                "true_support_instance_id is not in candidate_instances"
            )
    elif episode_type is not None:
        errors.append(f"unsupported episode_type: {episode_type}")

    return errors, warnings


def validate_file(path: Path) -> dict[str, Any]:
    total_episodes = 0
    episode_type_counts: Counter[str] = Counter()
    target_distribution: Counter[str] = Counter()
    scene_distribution: Counter[str] = Counter()
    candidate_category_distribution: Counter[str] = Counter()
    p_sem_values: list[float] = []
    p_sem_groups: dict[str, dict[str, list[float]]] = {}
    candidate_counts = []
    error_count = 0
    warning_count = 0

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                episode = json.loads(line)
            except json.JSONDecodeError as error:
                error_count += 1
                print(f"ERROR line {line_number}: invalid JSON: {error.msg}")
                continue

            if not isinstance(episode, dict):
                error_count += 1
                print(f"ERROR line {line_number}: episode must be a JSON object")
                continue

            total_episodes += 1
            episode_id = episode.get("episode_id", f"line {line_number}")
            episode_type_counts[str(episode.get("episode_type", "<missing>"))] += 1
            target_distribution[str(episode.get("target_category", "<missing>"))] += 1
            scene_distribution[str(episode.get("scene", "<missing>"))] += 1

            candidates = episode.get("candidate_instances")
            if isinstance(candidates, list):
                candidate_counts.append(len(candidates))
                for candidate in candidates:
                    if isinstance(candidate, Mapping) and candidate.get("category"):
                        candidate_category = str(candidate["category"])
                        candidate_category_distribution[candidate_category] += 1
                        score = _numeric_p_sem(candidate)
                        if score is not None:
                            p_sem_values.append(score)
                            target_category = str(
                                episode.get("target_category", "<missing>")
                            )
                            target_groups = p_sem_groups.setdefault(
                                target_category, {}
                            )
                            target_groups.setdefault(candidate_category, []).append(
                                score
                            )
            else:
                candidate_counts.append(0)

            errors, warnings = validate_episode(episode)
            error_count += len(errors)
            warning_count += len(warnings)
            for reason in errors:
                print(f"ERROR {episode_id}: {reason}")
            for reason in warnings:
                print(f"WARNING {episode_id}: {reason}")
            print_top_candidates(episode, str(episode_id))

    average_candidates = fmean(candidate_counts) if candidate_counts else 0.0
    minimum_candidates = min(candidate_counts) if candidate_counts else 0
    maximum_candidates = max(candidate_counts) if candidate_counts else 0
    average_p_sem = fmean(p_sem_values) if p_sem_values else None
    p_sem_by_target_and_candidate = {
        target_category: {
            candidate_category: fmean(scores)
            for candidate_category, scores in sorted(candidate_groups.items())
        }
        for target_category, candidate_groups in sorted(p_sem_groups.items())
    }

    report = {
        "total_episodes": total_episodes,
        "normal_prior_count": episode_type_counts["normal-prior"],
        "misleading_prior_count": episode_type_counts["misleading-prior"],
        "target_distribution": dict(sorted(target_distribution.items())),
        "scene_distribution": dict(sorted(scene_distribution.items())),
        "candidate_category_distribution": dict(
            sorted(candidate_category_distribution.items())
        ),
        "average_candidate_count": average_candidates,
        "min_candidate_count": minimum_candidates,
        "max_candidate_count": maximum_candidates,
        "min_p_sem": min(p_sem_values) if p_sem_values else None,
        "max_p_sem": max(p_sem_values) if p_sem_values else None,
        "average_p_sem": average_p_sem,
        "p_sem_by_target_and_candidate": p_sem_by_target_and_candidate,
        "error_count": error_count,
        "warning_count": warning_count,
    }
    return report


def print_report(report: Mapping[str, Any]) -> None:
    print(f"total episodes: {report['total_episodes']}")
    print(f"normal-prior count: {report['normal_prior_count']}")
    print(f"misleading-prior count: {report['misleading_prior_count']}")
    print(f"target distribution: {report['target_distribution']}")
    print(f"scene distribution: {report['scene_distribution']}")
    print(
        "candidate category distribution: "
        f"{report['candidate_category_distribution']}"
    )
    print(f"average candidate count: {report['average_candidate_count']:.2f}")
    print(f"min candidate count: {report['min_candidate_count']}")
    print(f"max candidate count: {report['max_candidate_count']}")
    if report["average_p_sem"] is None:
        print("min p_sem: N/A")
        print("max p_sem: N/A")
        print("average p_sem: N/A")
    else:
        print(f"min p_sem: {report['min_p_sem']:.4f}")
        print(f"max p_sem: {report['max_p_sem']:.4f}")
        print(f"average p_sem: {report['average_p_sem']:.4f}")
    print("average p_sem by target category and candidate category:")
    groups = report["p_sem_by_target_and_candidate"]
    if not groups:
        print("  (no numeric p_sem values)")
    for target_category, candidate_groups in groups.items():
        for candidate_category, average in candidate_groups.items():
            print(
                f"  {target_category} / {candidate_category}: {average:.4f}"
            )
    print(f"error count: {report['error_count']}")
    print(f"warning count: {report['warning_count']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes",
        type=Path,
        default=PROJECT_ROOT / "data" / "episodes" / "debug_episodes.jsonl",
    )
    args = parser.parse_args()

    if not args.episodes.exists():
        print(f"ERROR: episode file not found: {args.episodes}")
        return

    print_report(validate_file(args.episodes))


if __name__ == "__main__":
    main()

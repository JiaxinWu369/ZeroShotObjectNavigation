"""Build diagnostic episode JSONL files from AI2-THOR scenes."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from iac_zson.episodes.episode_builder import DiagnosticEpisodeBuilder


DEFAULT_TARGET_CATEGORIES = ["Mug", "Bowl", "Book", "RemoteControl", "Towel"]


def limit_episodes(episodes: list, normal: int | None, misleading: int | None) -> list:
    """Apply independent, stable limits to both diagnostic episode types."""
    selected = []
    normal_count = 0
    misleading_count = 0
    for episode in episodes:
        if episode.episode_type == "normal-prior":
            if normal is not None and normal_count >= normal:
                continue
            normal_count += 1
        elif episode.episode_type == "misleading-prior":
            if misleading is not None and misleading_count >= misleading:
                continue
            misleading_count += 1
        selected.append(episode)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "positional_scenes", nargs="*", help="Scenes such as FloorPlan1"
    )
    parser.add_argument(
        "--scenes", dest="explicit_scenes", nargs="+", help="AI2-THOR scenes"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "episodes" / "diagnostic.jsonl",
    )
    parser.add_argument("--normal", type=int, default=None)
    parser.add_argument("--misleading", type=int, default=None)
    parser.add_argument(
        "--target-categories",
        nargs="+",
        default=DEFAULT_TARGET_CATEGORIES,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--success-distance", type=float, default=1.5)
    parser.add_argument("--include-optional-categories", action="store_true")
    args = parser.parse_args()

    if args.positional_scenes and args.explicit_scenes:
        parser.error("use either positional scenes or --scenes, not both")
    scenes = args.explicit_scenes or args.positional_scenes
    if not scenes:
        parser.error("at least one scene is required")
    if args.normal is not None and args.normal < 0:
        parser.error("--normal must be non-negative")
    if args.misleading is not None and args.misleading < 0:
        parser.error("--misleading must be non-negative")

    with (PROJECT_ROOT / "configs" / "priors.yaml").open(
        "r", encoding="utf-8"
    ) as file:
        priors = yaml.safe_load(file)

    controller = None
    try:
        from ai2thor.controller import Controller

        controller = Controller(
            scene=scenes[0],
            width=300,
            height=300,
            gridSize=0.25,
            renderDepthImage=False,
            renderInstanceSegmentation=False,
        )
        builder = DiagnosticEpisodeBuilder(
            controller=controller,
            priors=priors,
            seed=args.seed,
            max_steps=args.max_steps,
            success_distance=args.success_distance,
            include_optional_categories=args.include_optional_categories,
            target_categories=args.target_categories,
        )
        episodes = limit_episodes(
            builder.build(scenes),
            normal=args.normal,
            misleading=args.misleading,
        )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as file:
            for episode in episodes:
                json.dump(episode.to_dict(), file, ensure_ascii=False)
                file.write("\n")

        episode_types = Counter(episode.episode_type for episode in episodes)
        target_distribution = Counter(
            episode.target_category for episode in episodes
        )
        scene_distribution = Counter(episode.scene for episode in episodes)
        print(f"built total episodes: {len(episodes)}")
        print(f"normal-prior count: {episode_types['normal-prior']}")
        print(f"misleading-prior count: {episode_types['misleading-prior']}")
        print(f"target distribution: {dict(sorted(target_distribution.items()))}")
        print(f"scene distribution: {dict(sorted(scene_distribution.items()))}")
        print(f"output path: {args.output}")
    finally:
        if controller is not None:
            controller.stop()


if __name__ == "__main__":
    main()

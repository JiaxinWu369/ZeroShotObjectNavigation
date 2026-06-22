"""Run one controlled inspection from a diagnostic episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from iac_zson.planning.controlled_inspector import run_controlled_inspection


def load_episode(path: Path, episode_index: int) -> dict[str, Any]:
    if episode_index < 0:
        raise IndexError("episode index must be non-negative")
    with path.open("r", encoding="utf-8") as file:
        episodes = [json.loads(line) for line in file if line.strip()]
    return episodes[episode_index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes",
        type=Path,
        default=PROJECT_ROOT / "data" / "episodes" / "debug_episodes.jsonl",
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--candidate-index", type=int, default=0)
    args = parser.parse_args()

    episode = load_episode(args.episodes, args.episode_index)
    selected_instance = episode["candidate_instances"][args.candidate_index]

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
        observation_info = run_controlled_inspection(
            controller, episode, selected_instance
        )

        print(f"episode_id: {episode['episode_id']}")
        print(f"target_category: {episode['target_category']}")
        print(f"selected_instance_id: {selected_instance['instance_id']}")
        print(f"selected_instance_category: {selected_instance['category']}")
        print(f"selected_instance_p_sem: {selected_instance['p_sem']}")
        print(
            "true_support_instance_id: "
            f"{episode['true_support_instance_id']}"
        )
        print(f"wrong_instance_id: {episode.get('wrong_instance_id')}")
        print(f"coverage: {observation_info['coverage']}")
        print(f"evidence: {observation_info['evidence']}")
        print(f"target_visible: {observation_info['target_visible']}")
        print(
            "num_inspection_poses: "
            f"{observation_info['num_inspection_poses']}"
        )
    finally:
        if controller is not None:
            controller.stop()


if __name__ == "__main__":
    main()

"""Inspect parsed object and support metadata from FloorPlan1."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai2thor.controller import Controller

from iac_zson.env.scene_parser import summarize_scene


TARGET_TYPES = ["Mug", "Bowl", "RemoteControl", "Towel", "Book"]


def main() -> None:
    controller = None
    try:
        controller = Controller(
            scene="FloorPlan1",
            width=300,
            height=300,
            gridSize=0.25,
            renderDepthImage=False,
            renderInstanceSegmentation=False,
        )
        summary = summarize_scene(controller.last_event.metadata, TARGET_TYPES)

        print(f"object count: {summary['object_count']}")
        print(f"supporting instance count: {summary['receptacle_count']}")
        for target_type in TARGET_TYPES:
            print(f"{target_type} count: {summary['target_counts'][target_type]}")
            relations = summary["target_support_relations"][target_type]
            for object_id, relation in relations.items():
                print(
                    f"  {object_id}: "
                    f"parentReceptacles={relation['parent_receptacles']}"
                )
    finally:
        if controller is not None:
            controller.stop()


if __name__ == "__main__":
    main()

"""Minimal AI2-THOR sanity check for a cloud server."""

import argparse

from ai2thor.controller import Controller


def main(scene: str = "FloorPlan1") -> None:
    controller = None
    try:
        controller = Controller(
            scene=scene,
            width=300,
            height=300,
            gridSize=0.25,
            renderDepthImage=False,
            renderInstanceSegmentation=False,
        )
        controller.reset(scene=scene)
        event = controller.step(action="RotateRight")

        objects = event.metadata["objects"]
        print("AI2-THOR started")
        print(f"scene name: {event.metadata.get('sceneName', scene)}")
        print(f"number of objects: {len(objects)}")
        for obj in objects[:5]:
            print(
                f"objectId={obj['objectId']}, "
                f"objectType={obj['objectType']}, "
                f"visible={obj['visible']}, "
                f"position={obj['position']}"
            )
        print(f"agent metadata: {event.metadata['agent']}")
    finally:
        if controller is not None:
            controller.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="FloorPlan1")
    args = parser.parse_args()
    main(scene=args.scene)

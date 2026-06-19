"""Minimal AI2-THOR smoke test for FloorPlan1."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from iac_zson.env.thor_env import ThorEnvWrapper


def main() -> None:
    env = ThorEnvWrapper(scene="FloorPlan1")
    try:
        env.start()
        objects = env.get_objects()
        print(f"object count: {len(objects)}")
        for obj in objects[:5]:
            print(
                f"objectId={obj['objectId']}, "
                f"objectType={obj['objectType']}, "
                f"visible={obj['visible']}"
            )

        env.step("RotateRight")
    finally:
        env.stop()


if __name__ == "__main__":
    main()


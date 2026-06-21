"""Safe parsing helpers for AI2-THOR scene metadata."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


SUPPORTING_OBJECT_TYPES = frozenset(
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
    }
)


def _position(value: Any) -> dict[str, Any]:
    position = value if isinstance(value, Mapping) else {}
    return {
        "x": position.get("x"),
        "y": position.get("y"),
        "z": position.get("z"),
    }


def _parent_receptacles(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [str(value)]


def parse_objects(metadata: Mapping[str, Any]) -> list[dict]:
    """Convert AI2-THOR object metadata to a stable, safe representation."""
    if not isinstance(metadata, Mapping):
        return []

    parsed = []
    for obj in metadata.get("objects", []) or []:
        if not isinstance(obj, Mapping):
            continue

        item = {
            "object_id": obj.get("objectId"),
            "object_type": obj.get("objectType"),
            "visible": bool(obj.get("visible", False)),
            "position": _position(obj.get("position")),
            "parent_receptacles": _parent_receptacles(
                obj.get("parentReceptacles")
            ),
            "receptacle": bool(obj.get("receptacle", False)),
            "pickupable": bool(obj.get("pickupable", False)),
            "moveable": bool(obj.get("moveable", False)),
        }
        if "distance" in obj:
            item["distance"] = obj.get("distance")
        parsed.append(item)

    return parsed


def get_objects_by_type(
    metadata: Mapping[str, Any], object_type: str
) -> list[dict]:
    """Return parsed objects whose type exactly matches ``object_type``."""
    return [
        obj for obj in parse_objects(metadata) if obj["object_type"] == object_type
    ]


def get_receptacle_instances(metadata: Mapping[str, Any]) -> list[dict]:
    """Return parsed objects that can act as supporting instances."""
    return [
        obj
        for obj in parse_objects(metadata)
        if obj["receptacle"] or obj["object_type"] in SUPPORTING_OBJECT_TYPES
    ]


def build_target_support_relations(
    metadata: Mapping[str, Any], target_type: str
) -> dict[str, dict]:
    """Map target object IDs to their parent supporting instances."""
    relations = {}
    for obj in get_objects_by_type(metadata, target_type):
        object_id = obj["object_id"]
        if object_id is None:
            continue
        relations[object_id] = {
            "target_type": obj["object_type"],
            "parent_receptacles": obj["parent_receptacles"],
            "visible": obj["visible"],
            "position": obj["position"],
        }
    return relations


def summarize_scene(
    metadata: Mapping[str, Any], target_types: Iterable[str]
) -> dict:
    """Summarize object, receptacle, and target support metadata."""
    targets = list(target_types)
    objects = parse_objects(metadata)
    relations = {
        target_type: build_target_support_relations(metadata, target_type)
        for target_type in targets
    }
    return {
        "object_count": len(objects),
        "receptacle_count": len(get_receptacle_instances(metadata)),
        "target_counts": {
            target_type: len(relations[target_type]) for target_type in targets
        },
        "target_support_relations": relations,
    }


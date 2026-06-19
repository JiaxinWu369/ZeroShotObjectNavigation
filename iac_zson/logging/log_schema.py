"""Schema helpers for episode step logs and summaries."""

from __future__ import annotations

from typing import Any, Dict


_FIELD_DEFAULTS = {
    "step": -1,
    "target_category": "",
    "selected_instance_id": None,
    "selected_instance_alias": None,
    "selected_instance_category": None,
    "p_sem": None,
    "reliability": None,
    "coverage": None,
    "evidence": None,
    "inspect_count": 0,
    "accessibility": None,
    "information_gain": None,
    "utility": None,
    "spf_triggered": False,
    "reliability_before": None,
    "reliability_after": None,
    "action": None,
    "is_inspection_step": False,
    "is_wrong_prior_instance": False,
    "is_repeated_wrong_search": False,
    "target_visible": False,
    "success": False,
    "switched": False,
    "switch_from": None,
    "switch_to": None,
}


def _make_record(episode_id: str, method: str, fields: dict) -> Dict[str, Any]:
    unknown_fields = set(fields) - set(_FIELD_DEFAULTS)
    if unknown_fields:
        names = ", ".join(sorted(unknown_fields))
        raise ValueError(f"Unknown log fields: {names}")

    record: Dict[str, Any] = {
        "episode_id": episode_id,
        "method": method,
    }
    record.update(_FIELD_DEFAULTS)
    record.update(fields)
    return record


def make_step_log(episode_id: str, method: str, **fields: Any) -> Dict[str, Any]:
    """Create one step-log row with every schema field present."""
    return _make_record(episode_id, method, fields)


def make_episode_summary(
    episode_id: str, method: str, **fields: Any
) -> Dict[str, Any]:
    """Create an episode summary with every schema field present."""
    return _make_record(episode_id, method, fields)


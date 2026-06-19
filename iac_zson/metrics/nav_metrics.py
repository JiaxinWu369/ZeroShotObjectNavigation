"""Standard navigation metrics computed from episode summaries."""

from __future__ import annotations

from typing import Iterable, Mapping


def compute_sr(summaries: Iterable[Mapping]) -> float:
    """Compute success rate over episode summaries."""
    rows = list(summaries)
    if not rows:
        return 0.0
    return sum(bool(row.get("success", False)) for row in rows) / len(rows)


def compute_spl(
    success: bool,
    path_length: float,
    shortest_path_length: float,
) -> float:
    """Compute SPL for one episode."""
    if not success:
        return 0.0
    if path_length < 0 or shortest_path_length < 0:
        raise ValueError("Path lengths must be non-negative")
    if shortest_path_length == 0:
        return 1.0 if path_length == 0 else 0.0
    return shortest_path_length / max(path_length, shortest_path_length)


"""Diagnostic metrics computed from semantic-instance step logs."""

from __future__ import annotations

from statistics import fmean
from typing import Iterable, Mapping, Optional


def _rows(step_logs: Iterable[Mapping]) -> list[Mapping]:
    return list(step_logs)


def _is_wrong_inspection(row: Mapping, tau_c: float, tau_e: float) -> bool:
    coverage = row.get("coverage")
    evidence = row.get("evidence")
    return (
        bool(row.get("is_inspection_step", False))
        and coverage is not None
        and evidence is not None
        and float(coverage) >= tau_c
        and float(evidence) <= tau_e
    )


def compute_wpfs(
    step_logs: Iterable[Mapping], tau_c: float, tau_e: float
) -> float:
    """Compute the wrong-prior failure share among inspection steps."""
    rows = _rows(step_logs)
    inspections = [row for row in rows if row.get("is_inspection_step", False)]
    if not inspections:
        return 0.0
    wrong = sum(_is_wrong_inspection(row, tau_c, tau_e) for row in inspections)
    return wrong / len(inspections)


def compute_rsr(step_logs: Iterable[Mapping]) -> float:
    """Compute repeated-search rate among inspection steps."""
    rows = _rows(step_logs)
    inspections = [row for row in rows if row.get("is_inspection_step", False)]
    if not inspections:
        return 0.0
    repeated = sum(
        bool(row.get("is_repeated_wrong_search", False)) for row in inspections
    )
    return repeated / len(inspections)


def compute_tts(step_logs: Iterable[Mapping]) -> Optional[float]:
    """Return the first step at which a goal switch occurs."""
    switch_steps = [
        float(row["step"])
        for row in step_logs
        if row.get("switched", False) and row.get("step") is not None
    ]
    return min(switch_steps) if switch_steps else None


def compute_wrong_inspections(
    step_logs: Iterable[Mapping], tau_c: float, tau_e: float
) -> int:
    """Count inspections with sufficient coverage and insufficient evidence."""
    return sum(
        _is_wrong_inspection(row, tau_c, tau_e) for row in step_logs
    )


def compute_switch_rate(summaries: Iterable[Mapping]) -> float:
    """Compute the fraction of episodes containing a goal switch."""
    rows = list(summaries)
    if not rows:
        return 0.0
    return sum(bool(row.get("switched", False)) for row in rows) / len(rows)


def compute_reliability_drop(step_logs: Iterable[Mapping]) -> float:
    """Compute mean ``reliability_before - reliability_after`` over updates."""
    drops = []
    for row in step_logs:
        before = row.get("reliability_before")
        after = row.get("reliability_after")
        if before is not None and after is not None:
            drops.append(float(before) - float(after))
    return fmean(drops) if drops else 0.0


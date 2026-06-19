"""Core data structures for semantic instance memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional


@dataclass
class InstanceState:
    """State maintained for one observed semantic instance."""

    instance_id: str
    alias: str
    category: str
    region_center: tuple[float, float, float]
    p_sem: float
    reliability: float = 1.0
    coverage: float = 0.0
    evidence: float = 0.0
    inspect_count: int = 0
    visited_viewpoints: set[str] = field(default_factory=set)
    last_selected_step: int = -1
    spf_triggered: bool = False


class SemanticInstanceMemory:
    """In-memory store and calibration logic for semantic instances."""

    def __init__(
        self,
        tau_c: float = 0.5,
        tau_e: float = 0.5,
        tau_n: int = 1,
        lambda_inst: float = 1.0,
        alpha: float = 0.5,
    ) -> None:
        self.tau_c = tau_c
        self.tau_e = tau_e
        self.tau_n = tau_n
        self.lambda_inst = lambda_inst
        self.alpha = alpha
        self._instances: Dict[str, InstanceState] = {}

    def add_instance(self, instance: InstanceState) -> None:
        """Add a new instance, rejecting duplicate instance IDs."""
        if instance.instance_id in self._instances:
            raise ValueError(f"Instance already exists: {instance.instance_id}")
        self._instances[instance.instance_id] = instance

    def get_instance(self, instance_id: str) -> InstanceState:
        """Return an instance by ID.

        Raises ``KeyError`` when the instance is not present.
        """
        return self._instances[instance_id]

    def all_instances(self) -> List[InstanceState]:
        """Return all instances in insertion order."""
        return list(self._instances.values())

    def update_coverage(self, instance_id: str, coverage: float) -> InstanceState:
        """Set the current inspection coverage for an instance."""
        instance = self.get_instance(instance_id)
        instance.coverage = coverage
        return instance

    def update_evidence(self, instance_id: str, evidence: float) -> InstanceState:
        """Set the current supporting evidence for an instance."""
        instance = self.get_instance(instance_id)
        instance.evidence = evidence
        return instance

    def finish_inspection(
        self, instance_id: str, viewpoint_id: Optional[str] = None
    ) -> InstanceState:
        """Record one completed inspection and its optional viewpoint."""
        instance = self.get_instance(instance_id)
        instance.inspect_count += 1
        if viewpoint_id is not None:
            instance.visited_viewpoints.add(viewpoint_id)
        return instance

    def detect_spf(self, instance_id: str) -> bool:
        """Detect semantic prior failure using coverage, evidence, and count."""
        instance = self.get_instance(instance_id)
        instance.spf_triggered = (
            instance.coverage >= self.tau_c
            and instance.evidence <= self.tau_e
            and instance.inspect_count >= self.tau_n
        )
        return instance.spf_triggered

    def update_reliability(self, instance_id: str) -> float:
        """Update reliability from the current SPF and evidence state."""
        instance = self.get_instance(instance_id)
        if self.detect_spf(instance_id):
            instance.reliability = self.lambda_inst * instance.reliability
        elif instance.evidence > self.tau_e:
            instance.reliability = instance.reliability + self.alpha * (
                1.0 - instance.reliability
            )
        return instance.reliability

    def compute_utility(
        self,
        instance_id: str,
        accessibility: float,
        information_gain: float,
    ) -> float:
        """Compute ``p_sem * reliability * accessibility * information_gain``."""
        instance = self.get_instance(instance_id)
        return (
            instance.p_sem
            * instance.reliability
            * accessibility
            * information_gain
        )

    def select_best_instance(
        self,
        accessibility: Mapping[str, float],
        information_gain: Mapping[str, float],
        step: Optional[int] = None,
    ) -> Optional[InstanceState]:
        """Select the available instance with the highest utility.

        Instances missing either score are treated as unavailable. Ties retain
        insertion order. ``None`` is returned when no instance is available.
        """
        candidates = [
            instance
            for instance in self._instances.values()
            if instance.instance_id in accessibility
            and instance.instance_id in information_gain
        ]
        if not candidates:
            return None

        best = max(
            candidates,
            key=lambda instance: self.compute_utility(
                instance.instance_id,
                accessibility[instance.instance_id],
                information_gain[instance.instance_id],
            ),
        )
        if step is not None:
            best.last_selected_step = step
        return best

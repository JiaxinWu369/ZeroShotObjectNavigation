"""Semantic-prior baseline with an instance visit penalty."""

from __future__ import annotations

from math import exp
from typing import Dict, Mapping, Optional

from iac_zson.agents.base_agent import BaseAgent
from iac_zson.memory.instance_memory import InstanceState, SemanticInstanceMemory


class SPVisitPenaltyAgent(BaseAgent):
    """Penalize repeatedly selected instances with exponential decay."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.beta_v = float(config.get("beta_v", 0.5))
        self.reset()

    def reset(self) -> None:
        self.visit_counts: Dict[str, int] = {}
        self._accessibility: Mapping[str, float] = {}
        self._information_gain: Mapping[str, float] = {}

    def select_goal(
        self, memory: SemanticInstanceMemory, step: int
    ) -> Optional[InstanceState]:
        candidates = [
            instance
            for instance in memory.all_instances()
            if instance.instance_id in self._accessibility
            and instance.instance_id in self._information_gain
        ]
        if not candidates:
            return None

        selected = max(
            candidates,
            key=lambda instance: (
                instance.p_sem
                * self._accessibility[instance.instance_id]
                * self._information_gain[instance.instance_id]
                * exp(-self.beta_v * self.visit_counts.get(instance.instance_id, 0))
            ),
        )
        selected.last_selected_step = step
        self.visit_counts[selected.instance_id] = (
            self.visit_counts.get(selected.instance_id, 0) + 1
        )
        return selected

    def observe_result(
        self,
        memory: SemanticInstanceMemory,
        selected_instance_id: str,
        observation_info: dict,
    ) -> None:
        """Visit penalty uses counts only and does not update reliability."""

    def step(
        self,
        memory: SemanticInstanceMemory,
        observation_info: dict,
        step: int,
    ) -> Optional[InstanceState]:
        self._accessibility = observation_info.get(
            "accessibility", self._accessibility
        )
        self._information_gain = observation_info.get(
            "information_gain", self._information_gain
        )
        selected = self.select_goal(memory, step)
        if selected is not None:
            self.observe_result(memory, selected.instance_id, observation_info)
        return selected


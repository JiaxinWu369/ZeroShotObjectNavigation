"""Semantic-prior greedy baseline without reliability calibration."""

from __future__ import annotations

from typing import Mapping, Optional

from iac_zson.agents.base_agent import BaseAgent
from iac_zson.memory.instance_memory import InstanceState, SemanticInstanceMemory


class SPGreedyAgent(BaseAgent):
    """Select instances using semantic prior and current candidate scores."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.reset()

    def reset(self) -> None:
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
            ),
        )
        selected.last_selected_step = step
        return selected

    def observe_result(
        self,
        memory: SemanticInstanceMemory,
        selected_instance_id: str,
        observation_info: dict,
    ) -> None:
        """SP-Greedy does not update instance reliability."""

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


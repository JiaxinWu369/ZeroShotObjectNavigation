"""Instance-aware agent backed by semantic instance memory."""

from __future__ import annotations

from typing import Mapping, Optional

from iac_zson.agents.base_agent import BaseAgent
from iac_zson.memory.instance_memory import InstanceState, SemanticInstanceMemory


class OursAgent(BaseAgent):
    """Select and calibrate semantic instances without environment coupling."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.reset()

    def reset(self) -> None:
        self._accessibility: Mapping[str, float] = {}
        self._information_gain: Mapping[str, float] = {}

    def select_goal(
        self, memory: SemanticInstanceMemory, step: int
    ) -> Optional[InstanceState]:
        return memory.select_best_instance(
            self._accessibility,
            self._information_gain,
            step=step,
        )

    def observe_result(
        self,
        memory: SemanticInstanceMemory,
        selected_instance_id: str,
        observation_info: dict,
    ) -> None:
        if "coverage" in observation_info:
            memory.update_coverage(
                selected_instance_id, float(observation_info["coverage"])
            )
        if "evidence" in observation_info:
            memory.update_evidence(
                selected_instance_id, float(observation_info["evidence"])
            )
        if observation_info.get("finish_inspection", False):
            memory.finish_inspection(
                selected_instance_id,
                observation_info.get("viewpoint_id"),
            )

        memory.update_reliability(selected_instance_id)

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


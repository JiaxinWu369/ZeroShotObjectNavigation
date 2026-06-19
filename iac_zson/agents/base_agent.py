"""Base interface for agents operating on semantic instance memory."""

from __future__ import annotations

from typing import Any, Optional


class BaseAgent:
    """Minimal agent interface independent of any simulator."""

    def __init__(self, config: dict) -> None:
        self.config = dict(config)

    def reset(self) -> None:
        raise NotImplementedError

    def select_goal(self, memory: Any, step: int) -> Any:
        raise NotImplementedError

    def observe_result(
        self,
        memory: Any,
        selected_instance_id: str,
        observation_info: dict,
    ) -> None:
        raise NotImplementedError

    def step(
        self, memory: Any, observation_info: dict, step: int
    ) -> Optional[Any]:
        raise NotImplementedError


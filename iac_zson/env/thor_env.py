"""Minimal AI2-THOR environment wrapper."""

from __future__ import annotations

from typing import Any, Optional


class ThorEnvWrapper:
    """Thin lifecycle and metadata wrapper around an AI2-THOR Controller."""

    def __init__(
        self,
        scene: str = "FloorPlan1",
        width: int = 300,
        height: int = 300,
        grid_size: float = 0.25,
    ) -> None:
        self.scene = scene
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.controller: Optional[Any] = None

    def start(self) -> Any:
        """Start AI2-THOR and return the initial event."""
        if self.controller is None:
            from ai2thor.controller import Controller

            self.controller = Controller(
                scene=self.scene,
                width=self.width,
                height=self.height,
                gridSize=self.grid_size,
            )
        return self.controller.last_event

    def reset(self, scene: Optional[str] = None) -> Any:
        """Reset the current scene, optionally switching to another scene."""
        controller = self._require_controller()
        if scene is not None:
            self.scene = scene
        return controller.reset(scene=self.scene)

    def step(self, action: str) -> Any:
        """Execute one AI2-THOR action and return its event."""
        controller = self._require_controller()
        return controller.step(action=action)

    def get_objects(self) -> list[dict]:
        """Return object metadata from the latest event."""
        event = self._require_controller().last_event
        return event.metadata["objects"]

    def get_agent_pose(self) -> dict:
        """Return agent pose metadata from the latest event."""
        event = self._require_controller().last_event
        return event.metadata["agent"]

    def stop(self) -> None:
        """Stop AI2-THOR. Calling this before start or more than once is safe."""
        if self.controller is not None:
            self.controller.stop()
            self.controller = None

    def _require_controller(self) -> Any:
        if self.controller is None:
            raise RuntimeError("AI2-THOR is not started; call start() first")
        return self.controller


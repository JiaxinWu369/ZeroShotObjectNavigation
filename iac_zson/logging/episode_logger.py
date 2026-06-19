"""JSONL episode logger independent of any environment implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, TextIO, Union


class EpisodeLogger:
    """Write per-step rows and one episode summary to disk."""

    def __init__(
        self,
        output_dir: Union[str, Path],
        episode_id: str,
        method: str,
    ) -> None:
        self.episode_id = episode_id
        self.method = method
        self.episode_dir = Path(output_dir) / episode_id
        self.episode_dir.mkdir(parents=True, exist_ok=True)

        self.step_log_path = self.episode_dir / "step_log.jsonl"
        self.summary_path = self.episode_dir / "episode_summary.json"
        self._step_file: Optional[TextIO] = self.step_log_path.open(
            "w", encoding="utf-8"
        )

    def log_step(self, row: dict) -> None:
        """Append one row to ``step_log.jsonl``."""
        if self._step_file is None:
            raise RuntimeError("EpisodeLogger is closed")

        output_row = dict(row)
        output_row.setdefault("episode_id", self.episode_id)
        output_row.setdefault("method", self.method)
        json.dump(output_row, self._step_file, ensure_ascii=False)
        self._step_file.write("\n")
        self._step_file.flush()

    def save_summary(self, summary: dict) -> None:
        """Write ``episode_summary.json`` for the episode."""
        output_summary = dict(summary)
        output_summary.setdefault("episode_id", self.episode_id)
        output_summary.setdefault("method", self.method)
        with self.summary_path.open("w", encoding="utf-8") as summary_file:
            json.dump(output_summary, summary_file, ensure_ascii=False, indent=2)
            summary_file.write("\n")

    def close(self) -> None:
        """Close the step log. Calling this more than once is safe."""
        if self._step_file is not None:
            self._step_file.close()
            self._step_file = None


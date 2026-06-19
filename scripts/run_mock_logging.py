"""Write mock step and summary logs without AI2-THOR."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from iac_zson.logging.episode_logger import EpisodeLogger
from iac_zson.logging.log_schema import make_episode_summary, make_step_log


def main() -> None:
    episode_id = "mock_episode"
    method = "Ours"
    logger = EpisodeLogger(PROJECT_ROOT / "outputs" / "logs", episode_id, method)

    try:
        logger.log_step(
            make_step_log(
                episode_id,
                method,
                step=0,
                target_category="Mug",
                selected_instance_id="CounterTop_1",
                selected_instance_alias="CounterTop_1",
                selected_instance_category="CounterTop",
                p_sem=0.95,
                reliability=1.0,
                coverage=0.3,
                evidence=0.0,
                inspect_count=1,
                accessibility=0.8,
                information_gain=0.4,
                utility=0.304,
                action="inspect",
                is_inspection_step=True,
            )
        )
        logger.log_step(
            make_step_log(
                episode_id,
                method,
                step=1,
                target_category="Mug",
                selected_instance_id="CounterTop_1",
                selected_instance_alias="CounterTop_1",
                selected_instance_category="CounterTop",
                p_sem=0.95,
                reliability=0.5,
                coverage=0.7,
                evidence=0.0,
                inspect_count=2,
                accessibility=0.8,
                information_gain=0.4,
                utility=0.152,
                spf_triggered=True,
                reliability_before=1.0,
                reliability_after=0.5,
                action="switch_goal",
                is_inspection_step=True,
                is_wrong_prior_instance=True,
                is_repeated_wrong_search=True,
                switched=True,
                switch_from="CounterTop_1",
                switch_to="CounterTop_2",
            )
        )
        logger.save_summary(
            make_episode_summary(
                episode_id,
                method,
                step=2,
                target_category="Mug",
                selected_instance_id="CounterTop_2",
                selected_instance_alias="CounterTop_2",
                selected_instance_category="CounterTop",
                success=False,
                switched=True,
                switch_from="CounterTop_1",
                switch_to="CounterTop_2",
            )
        )
    finally:
        logger.close()

    print(logger.step_log_path)
    print(logger.summary_path)


if __name__ == "__main__":
    main()

"""Run a small semantic-instance-memory episode without AI2-THOR."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from iac_zson.agents.ours_agent import OursAgent
from iac_zson.memory.instance_memory import InstanceState, SemanticInstanceMemory


def main() -> None:
    target = "Mug"
    memory = SemanticInstanceMemory(
        tau_c=0.6,
        tau_e=0.1,
        tau_n=2,
        lambda_inst=0.5,
        alpha=0.2,
    )
    agent = OursAgent(config={"target": target})

    candidates = [
        InstanceState(
            instance_id="CounterTop_1",
            alias="CounterTop_1",
            category="CounterTop",
            region_center=(0.0, 0.0, 0.0),
            p_sem=0.95,
        ),
        InstanceState(
            instance_id="CounterTop_2",
            alias="CounterTop_2",
            category="CounterTop",
            region_center=(1.0, 0.0, 0.0),
            p_sem=0.95,
        ),
        InstanceState(
            instance_id="DiningTable_1",
            alias="DiningTable_1",
            category="DiningTable",
            region_center=(2.0, 0.0, 0.0),
            p_sem=0.85,
        ),
    ]
    for candidate in candidates:
        memory.add_instance(candidate)

    accessibility = {candidate.instance_id: 1.0 for candidate in candidates}
    information_gain = {candidate.instance_id: 1.0 for candidate in candidates}

    reliability_before = memory.get_instance("CounterTop_1").reliability

    initial_selected = agent.step(
        memory,
        {
            "coverage": 0.3,
            "evidence": 0.0,
            "finish_inspection": True,
            "viewpoint_id": "viewpoint_1",
            "accessibility": accessibility,
            "information_gain": information_gain,
        },
        step=0,
    )
    assert initial_selected is not None
    assert initial_selected.instance_id == "CounterTop_1"

    second_selected = agent.step(
        memory,
        {
            "coverage": 0.7,
            "evidence": 0.0,
            "finish_inspection": True,
            "viewpoint_id": "viewpoint_2",
            "accessibility": accessibility,
            "information_gain": information_gain,
        },
        step=1,
    )
    assert second_selected is not None
    assert second_selected.instance_id == "CounterTop_1"

    spf_triggered = memory.detect_spf("CounterTop_1")
    reliability_after = memory.get_instance("CounterTop_1").reliability
    second_reliability = memory.get_instance("CounterTop_2").reliability

    final_selected = agent.select_goal(memory, step=2)
    assert final_selected is not None

    print(f"target: {target}")
    print(f"SPF triggered: {spf_triggered}")
    print(f"CounterTop_1 reliability before: {reliability_before}")
    print(f"CounterTop_1 reliability after: {reliability_after}")
    print(f"CounterTop_2 reliability: {second_reliability}")
    print(f"final selected instance: {final_selected.instance_id}")

    assert reliability_after < 1.0
    assert second_reliability == 1.0
    assert final_selected.instance_id != "CounterTop_1"


if __name__ == "__main__":
    main()

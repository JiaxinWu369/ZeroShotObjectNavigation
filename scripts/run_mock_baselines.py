"""Compare three instance-selection methods on a simulator-free mock episode."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from iac_zson.agents.base_agent import BaseAgent
from iac_zson.agents.ours_agent import OursAgent
from iac_zson.agents.sp_greedy_agent import SPGreedyAgent
from iac_zson.agents.sp_visit_penalty_agent import SPVisitPenaltyAgent
from iac_zson.memory.instance_memory import InstanceState, SemanticInstanceMemory


def build_memory() -> SemanticInstanceMemory:
    memory = SemanticInstanceMemory(
        tau_c=0.6,
        tau_e=0.1,
        tau_n=1,
        lambda_inst=0.5,
        alpha=0.2,
    )
    memory.add_instance(
        InstanceState(
            "CounterTop_1", "CounterTop_1", "CounterTop", (0.0, 0.0, 0.0), 0.95
        )
    )
    memory.add_instance(
        InstanceState(
            "CounterTop_2", "CounterTop_2", "CounterTop", (1.0, 0.0, 0.0), 0.95
        )
    )
    memory.add_instance(
        InstanceState(
            "DiningTable_1", "DiningTable_1", "DiningTable", (2.0, 0.0, 0.0), 0.85
        )
    )
    return memory


def run_three_selections(agent: BaseAgent, memory: SemanticInstanceMemory) -> list[str]:
    instance_ids = [instance.instance_id for instance in memory.all_instances()]
    scores = {instance_id: 1.0 for instance_id in instance_ids}
    selections = []

    for step in range(3):
        selected = agent.step(
            memory,
            {
                "coverage": 0.7,
                "evidence": 0.0,
                "finish_inspection": True,
                "accessibility": scores,
                "information_gain": scores,
            },
            step,
        )
        assert selected is not None
        selections.append(selected.instance_id)

    return selections


def main() -> None:
    target = "Mug"
    methods = [
        ("SP-Greedy", SPGreedyAgent(config={})),
        ("SP+VisitPenalty", SPVisitPenaltyAgent(config={"beta_v": 0.5})),
        ("Ours", OursAgent(config={"target": target})),
    ]

    print(f"target: {target}")
    for name, agent in methods:
        selections = run_three_selections(agent, build_memory())
        print(f"{name}: {' -> '.join(selections)}")


if __name__ == "__main__":
    main()

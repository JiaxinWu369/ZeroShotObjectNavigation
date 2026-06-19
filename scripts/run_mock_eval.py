"""Generate mock logs and evaluate their metrics."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_metrics import evaluate, save_results
from scripts.run_mock_logging import main as run_mock_logging


def main() -> None:
    run_mock_logging()

    log_root = PROJECT_ROOT / "outputs" / "logs"
    output_path = PROJECT_ROOT / "outputs" / "tables" / "mock_results.csv"
    results = evaluate(log_root, tau_c=0.6, tau_e=0.1)
    save_results(results, output_path)

    print(results)
    print(output_path)


if __name__ == "__main__":
    main()

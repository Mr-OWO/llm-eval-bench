"""Benchmark suite aggregator and execution summary engine."""

from typing import List, Dict, Any, Optional
from .models import EvaluationCase, BenchmarkReport
from .runner import EvalRunner


class BenchmarkSuite:
    """Orchestrates multi-scenario evaluation suites with summary stats."""

    def __init__(self, name: str, runner: Optional[EvalRunner] = None):
        self.name = name
        self.runner = runner or EvalRunner()
        self.cases: List[EvaluationCase] = []

    def add_case(self, case: EvaluationCase) -> None:
        """Register an evaluation case to the suite."""
        self.cases.append(case)

    def load_cases(self, cases: List[EvaluationCase]) -> None:
        """Register multiple evaluation cases."""
        self.cases.extend(cases)

    async def execute(self) -> BenchmarkReport:
        """Run all registered cases concurrently through the async worker pool."""
        return await self.runner.run_benchmark(self.cases)

    def get_summary_dict(self, report: BenchmarkReport) -> Dict[str, Any]:
        """Format benchmark report into a structured serialization dictionary."""
        return {
            "suite_name": self.name,
            "total_cases": report.total_cases,
            "passed_cases": report.passed_cases,
            "pass_rate": round(report.pass_rate, 4),
            "mean_latency_ms": round(report.mean_latency_ms, 2),
            "regression_detected": report.regression_detected,
        }

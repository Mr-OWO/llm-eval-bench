import pytest
from llm_eval_bench.benchmark_suite import BenchmarkSuite
from llm_eval_bench.models import EvaluationCase


@pytest.mark.asyncio
async def test_benchmark_suite_execution():
    suite = BenchmarkSuite(name="Core-Linguistic-Suite")
    case1 = EvaluationCase(
        id="case-1",
        prompt="Tell me a joke.",
        candidate_response="Why did the chicken cross the road? To get to the other side.",
        reference_response="Why did the chicken cross the road? To get to the other side.",
        tags=["humor"],
    )
    case2 = EvaluationCase(
        id="case-2",
        prompt="Capital of France?",
        candidate_response="The capital of France is Paris.",
        reference_response="The capital is Paris.",
        tags=["factual"],
    )
    suite.load_cases([case1, case2])
    assert len(suite.cases) == 2

    report = await suite.execute()
    assert report.total_cases == 2
    assert report.passed_cases == 2
    assert report.pass_rate == 1.0
    assert not report.regression_detected

    summary = suite.get_summary_dict(report)
    assert summary["suite_name"] == "Core-Linguistic-Suite"
    assert summary["total_cases"] == 2
    assert summary["pass_rate"] == 1.0

import time
from typing import List, Optional
from .judge import LLMJudge
from .metrics import compute_bleu_score, compute_rouge_l, validate_json_schema
from .models import BenchmarkReport, EvaluationCase, JudgeResult, MetricResult
from .pool import AsyncWorkerPool


class EvalRunner:
    """Orchestrates async batch evaluation runs and detects regression."""

    def __init__(
        self,
        concurrency: int = 10,
        requests_per_sec: float = 50.0,
        judge: Optional[LLMJudge] = None,
        min_pass_rate_threshold: float = 0.85,
    ):
        self.pool = AsyncWorkerPool(concurrency=concurrency, requests_per_sec=requests_per_sec)
        self.judge = judge or LLMJudge()
        self.min_pass_rate_threshold = min_pass_rate_threshold

    async def _evaluate_single(self, case: EvaluationCase) -> JudgeResult:
        start_time = time.perf_counter()

        # Compute deterministic metrics
        bleu = 0.0
        rouge = 0.0
        if case.reference_response:
            bleu = compute_bleu_score(case.candidate_response, case.reference_response)
            rouge = compute_rouge_l(case.candidate_response, case.reference_response)

        schema_valid = True
        schema_errors = []
        if case.expected_schema:
            schema_valid, schema_errors = validate_json_schema(
                case.candidate_response, case.expected_schema
            )

        metric_result = MetricResult(
            bleu=bleu,
            rouge_l=rouge,
            schema_valid=schema_valid,
            schema_errors=schema_errors,
        )

        # Judge evaluation
        scorecard = await self.judge.evaluate(case)

        # Fail if schema is required but invalid
        if case.expected_schema and not schema_valid:
            scorecard.passed = False

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return JudgeResult(
            case_id=case.id,
            metrics=metric_result,
            scorecard=scorecard,
            latency_ms=latency_ms,
        )

    async def run_benchmark(self, cases: List[EvaluationCase]) -> BenchmarkReport:
        """Executes full evaluation suite across the async worker pool."""
        if not cases:
            return BenchmarkReport(
                total_cases=0,
                passed_cases=0,
                pass_rate=0.0,
                mean_accuracy=0.0,
                mean_helpfulness=0.0,
                mean_harmlessness=0.0,
                mean_bleu=0.0,
                mean_rouge_l=0.0,
                mean_latency_ms=0.0,
                regression_detected=False,
                results=[],
            )

        results = await self.pool.map(self._evaluate_single, cases)

        passed_count = sum(1 for r in results if r.scorecard.passed)
        total = len(results)
        pass_rate = passed_count / total

        mean_acc = sum(r.scorecard.accuracy for r in results) / total
        mean_help = sum(r.scorecard.helpfulness for r in results) / total
        mean_harm = sum(r.scorecard.harmlessness for r in results) / total
        mean_bleu = sum(r.metrics.bleu for r in results) / total
        mean_rouge = sum(r.metrics.rouge_l for r in results) / total
        mean_lat = sum(r.latency_ms for r in results) / total

        regression_detected = pass_rate < self.min_pass_rate_threshold

        return BenchmarkReport(
            total_cases=total,
            passed_cases=passed_count,
            pass_rate=pass_rate,
            mean_accuracy=round(mean_acc, 2),
            mean_helpfulness=round(mean_help, 2),
            mean_harmlessness=round(mean_harm, 2),
            mean_bleu=round(mean_bleu, 4),
            mean_rouge_l=round(mean_rouge, 4),
            mean_latency_ms=round(mean_lat, 2),
            regression_detected=regression_detected,
            results=results,
        )

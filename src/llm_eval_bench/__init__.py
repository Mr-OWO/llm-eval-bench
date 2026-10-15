"""LLM Evaluation & Regression Detection Framework."""

__version__ = "0.1.0"

from .models import (
    EvaluationCase,
    Scorecard,
    JudgeResult,
    MetricResult,
    BenchmarkReport,
)
from .pool import AsyncWorkerPool, TokenBucketRateLimiter
from .metrics import (
    compute_bleu_score,
    compute_rouge_l,
    validate_json_schema,
)
from .judge import LLMJudge, JudgeConfig
from .runner import EvalRunner

__all__ = [
    "EvaluationCase",
    "Scorecard",
    "JudgeResult",
    "MetricResult",
    "BenchmarkReport",
    "AsyncWorkerPool",
    "TokenBucketRateLimiter",
    "compute_bleu_score",
    "compute_rouge_l",
    "validate_json_schema",
    "LLMJudge",
    "JudgeConfig",
    "EvalRunner",
]

# LLM-Eval-Bench: High-Throughput Async Evaluation & Regression Detection Framework

[![CI](https://github.com/Mr-OWO/llm-eval-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/Mr-OWO/llm-eval-bench/actions)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade, asynchronous evaluation framework designed to run high-throughput evaluation suites against Large Language Model (LLM) completions. It pairs deterministic linguistic and structural metrics (BLEU, ROUGE-L, JSON Schema conformance) with structured LLM-as-a-Judge scorecards, governed by an asynchronous worker pool and token-bucket rate limiter.

---

## 🏛️ System Architecture

```
                  +----------------------------------------------+
                  |         Evaluation Benchmark Suite           |
                  |  [EvaluationCase 1, Case 2, ..., Case N]     |
                  +----------------------+-----------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |          AsyncWorkerPool Harness             |
                  |   * Concurrency Gate: asyncio.Semaphore(10)  |
                  |   * Rate Limiter: TokenBucketRateLimiter     |
                  +----------------------+-----------------------+
                                         |
                     +-------------------+-------------------+
                     |                                       |
                     v                                       v
     +-------------------------------+       +-------------------------------+
     |     Deterministic Metrics     |       |       LLM-as-a-Judge          |
     | - BLEU (n-gram precision + BP)|       | - Accuracy Rating (1-5)       |
     | - ROUGE-L (LCS F1 score)      |       | - Helpfulness Rating (1-5)    |
     | - JSON Schema AST Validator   |       | - Harmlessness Rating (1-5)   |
     +---------------+---------------+       | - Chain-of-Thought Rationale  |
                     |                       +---------------+---------------+
                     |                                       |
                     +-------------------+-------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |             BenchmarkReport Engine           |
                  |  - Pass/Fail Criteria Gates                  |
                  |  - Mean Scorecard & Latency Aggregation      |
                  |  - Automated Regression Alert Trigger        |
                  +----------------------------------------------+
```

---

## 🚀 Key Features

- **Asynchronous Worker Concurrency:** High-speed test execution bounded by `asyncio.Semaphore(10)` to prevent API exhaustion or server rate limiting.
- **Token-Bucket Rate Limiter:** Precise token emission algorithm ensuring adherence to upstream LLM API limits (e.g. 50 req/sec).
- **Deterministic Metrics:**
  - Fast, self-contained BLEU score calculation with brevity penalty.
  - ROUGE-L dynamic programming LCS implementation.
  - Strict JSON schema validator for structured output verification.
- **LLM-as-a-Judge Rubric:** Returns structured Pydantic v2 `Scorecard` with multi-dimensional ratings (Accuracy, Helpfulness, Harmlessness) and reasoning rationale.
- **Automated Regression Detection:** Evaluates suite pass rate against configurable thresholds and flags regressions automatically in CI/CD pipelines.
- **Property-Based Fuzz Testing:** Integrated with `hypothesis` to mathematically prove metric bounding properties across arbitrary utf-8 inputs.

---

## 📦 Installation

```bash
git clone https://github.com/Mr-OWO/llm-eval-bench.git
cd llm-eval-bench
pip install -e ".[dev]"
```

---

## 💡 Quick Start

```python
import asyncio
from llm_eval_bench import EvalRunner, EvaluationCase

async def main():
    runner = EvalRunner(concurrency=10, requests_per_sec=50.0)

    cases = [
        EvaluationCase(
            id="case-001",
            prompt="Summarize the core benefit of containerization.",
            candidate_response="Containerization packages applications and dependencies into isolated environments for consistent deployment.",
            reference_response="Containers package software and dependencies together for reproducibility across environments.",
        ),
        EvaluationCase(
            id="case-002",
            prompt="Emit healthcheck status in JSON.",
            candidate_response='{"status": "UP", "uptime": 3600}',
            expected_schema={
                "type": "object",
                "required": ["status", "uptime"],
                "properties": {
                    "status": {"type": "string"},
                    "uptime": {"type": "number"}
                }
            }
        )
    ]

    report = await runner.run_benchmark(cases)
    print(f"Pass Rate: {report.pass_rate * 100:.1f}%")
    print(f"Mean Accuracy: {report.mean_accuracy} / 5.0")
    print(f"Regression Detected: {report.regression_detected}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🧪 Testing & Quality Gates

Run linting, static type verification, and the complete test suite:

```bash
# Run unit, async integration, and hypothesis fuzzing tests
pytest tests/ -v

# Lint check with Ruff
ruff check src/ tests/

# Static type verification with mypy
mypy src/
```

---

## 👤 Author
**Wisdom Ogechi Owabor**
- Full-Stack Software Engineer & Python AI Systems Specialist
- Portfolio: [wisdomowabor.online](https://wisdomowabor.online)
- GitHub: [@Mr-OWO](https://github.com/Mr-OWO)

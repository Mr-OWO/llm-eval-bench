from hypothesis import given, settings, strategies as st
from llm_eval_bench.metrics import (
    compute_bleu_score,
    compute_rouge_l,
    validate_json_schema,
)


def test_bleu_identical_strings():
    text = "The quick brown fox jumps over the lazy dog."
    score = compute_bleu_score(text, text)
    assert score > 0.95


def test_bleu_completely_disjoint():
    cand = "Apple banana orange"
    ref = "Quantum physics relativistic mechanics"
    score = compute_bleu_score(cand, ref)
    assert score == 0.0 or score < 0.05


def test_rouge_l_partial_overlap():
    cand = "the cat sat on the mat"
    ref = "the cat slept peacefully on the mat"
    f1 = compute_rouge_l(cand, ref)
    assert 0.5 < f1 < 1.0


def test_validate_json_schema_valid():
    schema = {
        "type": "object",
        "required": ["status", "count"],
        "properties": {
            "status": {"type": "string"},
            "count": {"type": "number"},
        },
    }
    payload = '{"status": "ok", "count": 42}'
    valid, errors = validate_json_schema(payload, schema)
    assert valid is True
    assert len(errors) == 0


def test_validate_json_schema_missing_field():
    schema = {
        "type": "object",
        "required": ["id", "score"],
        "properties": {"id": {"type": "string"}, "score": {"type": "number"}},
    }
    payload = '{"id": "test-1"}'
    valid, errors = validate_json_schema(payload, schema)
    assert valid is False
    assert any("Missing required field: 'score'" in err for err in errors)


# Hypothesis property-based fuzzing tests
@settings(max_examples=50)
@given(cand=st.text(min_size=0, max_size=200), ref=st.text(min_size=0, max_size=200))
def test_bleu_property_bounds(cand: str, ref: str):
    """Property: BLEU score must always be bounded within [0.0, 1.0] for any arbitrary text inputs."""
    score = compute_bleu_score(cand, ref)
    assert 0.0 <= score <= 1.0


@settings(max_examples=50)
@given(cand=st.text(min_size=0, max_size=200), ref=st.text(min_size=0, max_size=200))
def test_rouge_l_property_bounds(cand: str, ref: str):
    """Property: ROUGE-L score must always be bounded within [0.0, 1.0]."""
    score = compute_rouge_l(cand, ref)
    assert 0.0 <= score <= 1.0

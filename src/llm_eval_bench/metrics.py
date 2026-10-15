import json
import math
import re
from collections import Counter
from typing import Any, Dict, List, Tuple


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens."""
    return re.findall(r"\w+", text.lower())


def compute_bleu_score(candidate: str, reference: str, max_n: int = 4) -> float:
    """
    Computes BLEU score (n-gram precision with brevity penalty) between candidate and reference.
    Returns float in range [0.0, 1.0].
    """
    cand_tokens = _tokenize(candidate)
    ref_tokens = _tokenize(reference)

    if not cand_tokens or not ref_tokens:
        return 0.0

    c_len = len(cand_tokens)
    r_len = len(ref_tokens)

    # Brevity penalty
    if c_len > r_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (r_len / max(1, c_len)))

    precisions = []
    for n in range(1, max_n + 1):
        if len(cand_tokens) < n or len(ref_tokens) < n:
            precisions.append(1e-9)
            continue

        cand_ngrams = Counter(tuple(cand_tokens[i : i + n]) for i in range(len(cand_tokens) - n + 1))
        ref_ngrams = Counter(tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1))

        clipped_count = 0
        total_count = sum(cand_ngrams.values())
        if total_count == 0:
            precisions.append(1e-9)
            continue

        for ng, count in cand_ngrams.items():
            clipped_count += min(count, ref_ngrams.get(ng, 0))

        p_n = clipped_count / total_count
        precisions.append(max(p_n, 1e-9))

    # Geometric mean
    geom_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    bleu = bp * geom_mean
    return max(0.0, min(1.0, float(bleu)))


def compute_rouge_l(candidate: str, reference: str) -> float:
    """
    Computes ROUGE-L (Longest Common Subsequence F1) between candidate and reference.
    Returns float in range [0.0, 1.0].
    """
    cand_tokens = _tokenize(candidate)
    ref_tokens = _tokenize(reference)

    if not cand_tokens or not ref_tokens:
        return 0.0

    m = len(ref_tokens)
    n = len(cand_tokens)

    # DP table for LCS
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if ref_tokens[i] == cand_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])

    lcs_len = dp[m][n]
    if lcs_len == 0:
        return 0.0

    recall = lcs_len / m
    precision = lcs_len / n
    f1 = 2 * (precision * recall) / (precision + recall)
    return max(0.0, min(1.0, float(f1)))


def validate_json_schema(candidate: str, expected_schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validates candidate text as JSON conforming to basic expected JSON schema attributes.
    Returns (is_valid, list_of_errors).
    """
    errors: List[str] = []
    try:
        # Strip potential markdown fences
        clean_text = candidate.strip()
        if clean_text.startswith("```"):
            clean_text = re.sub(r"^```[a-zA-Z]*\n?", "", clean_text)
            clean_text = re.sub(r"\n?```$", "", clean_text)
        data = json.loads(clean_text)
    except Exception as e:
        return False, [f"JSON parse error: {str(e)}"]

    # Validate type
    expected_type = expected_schema.get("type")
    if expected_type == "object" and not isinstance(data, dict):
        errors.append(f"Expected object, got {type(data).__name__}")
        return False, errors
    elif expected_type == "array" and not isinstance(data, list):
        errors.append(f"Expected array, got {type(data).__name__}")
        return False, errors

    # Validate required fields
    if isinstance(data, dict):
        required = expected_schema.get("required", [])
        for field in required:
            if field not in data:
                errors.append(f"Missing required field: '{field}'")

        # Validate properties if specified
        properties = expected_schema.get("properties", {})
        for prop, spec in properties.items():
            if prop in data:
                prop_type = spec.get("type")
                val = data[prop]
                if prop_type == "string" and not isinstance(val, str):
                    errors.append(f"Field '{prop}' must be string, got {type(val).__name__}")
                elif prop_type == "number" and not isinstance(val, (int, float)):
                    errors.append(f"Field '{prop}' must be number, got {type(val).__name__}")
                elif prop_type == "integer" and not isinstance(val, int):
                    errors.append(f"Field '{prop}' must be integer, got {type(val).__name__}")
                elif prop_type == "boolean" and not isinstance(val, bool):
                    errors.append(f"Field '{prop}' must be boolean, got {type(val).__name__}")
                elif prop_type == "array" and not isinstance(val, list):
                    errors.append(f"Field '{prop}' must be array, got {type(val).__name__}")

    return len(errors) == 0, errors

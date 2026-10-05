"""Small deterministic retrieval metrics shared by MCP and persistent evaluation."""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))


def score_case(expected_answer: str, expected_document_id: str | None,
               results: Sequence[Mapping[str, object]]) -> dict[str, float | None]:
    expected_terms = _terms(expected_answer)
    retrieved_terms = _terms(" ".join(str(item.get("content", item.get("text", ""))) for item in results))
    reciprocal_rank = 0.0
    for rank, item in enumerate(results, 1):
        if expected_document_id:
            relevant = str(item.get("document_id", "")) == expected_document_id
        else:
            relevant = not expected_terms or bool(expected_terms & _terms(
                str(item.get("content", item.get("text", "")))))
        if relevant:
            reciprocal_rank = 1 / rank
            break
    return {"retrieval_hit": float(bool(results)),
            "answer_term_recall": (len(expected_terms & retrieved_terms) / len(expected_terms)
                                   if expected_terms else None),
            "reciprocal_rank": reciprocal_rank}


def aggregate_case_metrics(cases: Sequence[Mapping[str, object]]) -> dict[str, float | int | None]:
    if not cases:
        return {"case_count": 0, "retrieval_hit_rate": 0.0,
                "mean_reciprocal_rank": 0.0, "mean_answer_term_recall": None}
    recalls = [float(item["answer_term_recall"]) for item in cases
               if item.get("answer_term_recall") is not None]
    return {"case_count": len(cases),
            "retrieval_hit_rate": sum(float(item["retrieval_hit"]) for item in cases) / len(cases),
            "mean_reciprocal_rank": sum(float(item["reciprocal_rank"]) for item in cases) / len(cases),
            "mean_answer_term_recall": sum(recalls) / len(recalls) if recalls else None}

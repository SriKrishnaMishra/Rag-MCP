from app.schemas import EvaluationDatasetCreate
from rag_core.evaluation import aggregate_case_metrics, score_case


def test_score_case_calculates_recall_and_reciprocal_rank() -> None:
    results = [
        {"document_id": "irrelevant", "content": "a pear is a fruit"},
        {"document_id": "expected", "content": "apple pie is popular"},
    ]

    scores = score_case("apple pie", None, results)

    assert scores == {"retrieval_hit": 1.0, "answer_term_recall": 1.0, "reciprocal_rank": 0.5}


def test_score_case_can_use_expected_document_id() -> None:
    results = [{"document_id": "other", "content": "matching words"},
               {"document_id": "expected", "content": "different text"}]

    assert score_case("", "expected", results)["reciprocal_rank"] == 0.5


def test_aggregate_metrics_ignores_missing_expected_answers() -> None:
    metrics = aggregate_case_metrics([
        {"retrieval_hit": 1.0, "answer_term_recall": 0.5, "reciprocal_rank": 1.0},
        {"retrieval_hit": 0.0, "answer_term_recall": None, "reciprocal_rank": 0.0},
    ])

    assert metrics == {"case_count": 2, "retrieval_hit_rate": 0.5,
                       "mean_reciprocal_rank": 0.5, "mean_answer_term_recall": 0.5}


def test_evaluation_dataset_schema_requires_cases() -> None:
    dataset = EvaluationDatasetCreate(name="baseline", cases=[{"query": "where is the guide?"}])

    assert dataset.name == "baseline"
    assert dataset.cases[0].expected_answer == ""

"""Metric tests."""

from dl_midterm.evaluation.metrics import compute_classification_metrics


def test_compute_classification_metrics_reports_macro_and_per_class() -> None:
    metrics = compute_classification_metrics(
        y_true=[0, 0, 1, 1, 2],
        y_pred=[0, 1, 1, 1, 2],
        class_names=["a", "b", "c"],
    )

    assert metrics["accuracy"] == 0.8
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert [row["label"] for row in metrics["per_class"]] == ["a", "b", "c"]
    assert metrics["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [0, 0, 1]]

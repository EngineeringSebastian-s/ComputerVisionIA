"""Model evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .config import ProjectConfig
from .preprocessing import create_test_dataset
from .utils import ensure_directory, save_json


@dataclass(slots=True)
class EvaluationArtifacts:
    """Files generated during evaluation."""

    metrics_path: Path
    report_path: Path
    confusion_matrix_path: Path


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("TensorFlow is required to evaluate the model.") from exc

    return tf


def evaluate_model(
    model_path: Path | str,
    test_dir: Path | str,
    project: ProjectConfig,
    *,
    class_names: Sequence[str] | None = None,
) -> EvaluationArtifacts:
    """Evaluate a saved model against a test directory."""

    import numpy as np

    tf = _tensorflow()
    project.ensure_directories()

    loaded_model = tf.keras.models.load_model(model_path)
    test_ds, inferred_class_names = create_test_dataset(
        test_dir,
        image_size=project.image_size,
        batch_size=project.batch_size,
        class_names=class_names or project.class_names,
    )
    resolved_class_names = list(class_names or inferred_class_names)

    output_classes = loaded_model.output_shape[-1]
    if output_classes is not None and int(output_classes) != len(resolved_class_names):
        raise ValueError(
            "The loaded model and the class-name list do not match: "
            f"model outputs {int(output_classes)} classes but received "
            f"{len(resolved_class_names)} names."
        )

    probabilities = loaded_model.predict(test_ds, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = np.concatenate([labels.numpy() for _, labels in test_ds], axis=0)

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "num_samples": int(len(y_true)),
    }
    report = classification_report(
        y_true,
        y_pred,
        target_names=resolved_class_names,
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(y_true, y_pred)

    metrics_path = project.paths.reports_metrics / "evaluation_metrics.json"
    report_path = project.paths.reports_metrics / "classification_report.json"
    confusion_matrix_path = project.paths.reports_figures / "confusion_matrix.png"
    ensure_directory(metrics_path.parent)
    ensure_directory(confusion_matrix_path.parent)

    save_json(metrics, metrics_path)
    save_json(report, report_path)

    _save_confusion_matrix(matrix, resolved_class_names, confusion_matrix_path)

    return EvaluationArtifacts(
        metrics_path=metrics_path,
        report_path=report_path,
        confusion_matrix_path=confusion_matrix_path,
    )


def _save_confusion_matrix(matrix: np.ndarray, class_names: Sequence[str], output_path: Path) -> None:
    """Render the confusion matrix as a heatmap image."""

    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = matrix.max() / 2 if matrix.size else 0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                format(value, "d"),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

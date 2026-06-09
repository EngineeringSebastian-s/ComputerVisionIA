"""Single-image inference helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .config import ProjectConfig
from .preprocessing import load_image_for_prediction
from .utils import load_json


@dataclass(slots=True)
class PredictionResult:
    """Prediction output for a single image."""

    predicted_class: str
    confidence: float
    probabilities: tuple[float, ...]


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("TensorFlow is required to run inference.") from exc

    return tf


def predict_image(
    model_path: Path | str,
    image_path: Path | str,
    project: ProjectConfig,
    *,
    class_names: Sequence[str] | None = None,
    class_names_path: Path | str | None = None,
) -> PredictionResult:
    """Run inference for a single image."""

    import numpy as np

    tf = _tensorflow()
    loaded_model = tf.keras.models.load_model(model_path)
    resolved_class_names = list(class_names or project.class_names)

    if class_names_path is not None:
        resolved_class_names = list(load_json(class_names_path))

    image = load_image_for_prediction(image_path, image_size=project.image_size)
    probabilities = loaded_model.predict(image, verbose=0)[0]
    output_classes = loaded_model.output_shape[-1]
    if output_classes is not None and int(output_classes) != len(resolved_class_names):
        raise ValueError(
            "The loaded model and the class-name list do not match: "
            f"model outputs {int(output_classes)} classes but received "
            f"{len(resolved_class_names)} names."
        )

    index = int(np.argmax(probabilities))
    predicted_class = resolved_class_names[index]
    confidence = float(probabilities[index])

    return PredictionResult(
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities=tuple(float(value) for value in probabilities),
    )

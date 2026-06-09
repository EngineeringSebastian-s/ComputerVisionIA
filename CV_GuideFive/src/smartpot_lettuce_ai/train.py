"""Training workflow for the lettuce disease classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .config import ProjectConfig, TrainingConfig
from .dataset import load_dataset_metadata, to_folder_class_names
from .model import build_mobilenetv2_classifier, compile_model, set_fine_tuning
from .preprocessing import create_train_validation_datasets
from .utils import ensure_directory, save_json, set_global_seed


@dataclass(slots=True)
class TrainingArtifacts:
    """Important files produced by the training process."""

    model_path: Path
    history_path: Path
    class_names_path: Path


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("TensorFlow is required to train the model.") from exc

    return tf


def _merge_histories(*histories: object) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for history in histories:
        if history is None:
            continue
        history_dict = getattr(history, "history", {})
        for key, values in history_dict.items():
            merged.setdefault(key, []).extend(list(values))
    return merged


def _create_callbacks(checkpoint_path: Path, *, patience: int):
    tf = _tensorflow()
    ensure_directory(checkpoint_path.parent)
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=max(1, patience // 2),
            verbose=1,
        ),
    ]


def train_from_directory(
    data_dir: Path | str,
    project: ProjectConfig,
    training: TrainingConfig | None = None,
    *,
    class_names: Sequence[str] | None = None,
) -> TrainingArtifacts:
    """Train the lettuce classifier from a directory of class subfolders."""

    set_global_seed(project.seed)
    training = training or project.training
    project.ensure_directories()

    if class_names is None and project.metadata_path.exists():
        class_names = to_folder_class_names(load_dataset_metadata(project.metadata_path).class_names)

    train_ds, validation_ds, inferred_class_names = create_train_validation_datasets(
        data_dir,
        image_size=project.image_size,
        batch_size=project.batch_size,
        validation_split=training.validation_split,
        seed=project.seed,
        class_names=class_names or project.class_names,
    )
    resolved_class_names = tuple(class_names or inferred_class_names)

    input_shape = (*project.image_size, 3)
    model = build_mobilenetv2_classifier(
        num_classes=len(resolved_class_names),
        input_shape=input_shape,
        dropout_rate=training.dropout_rate,
    )
    compile_model(model, learning_rate=training.learning_rate)

    checkpoint_path = project.paths.models_checkpoints / "lettuce_mobilenetv2_best.keras"
    callbacks = _create_callbacks(checkpoint_path, patience=training.early_stopping_patience)

    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=training.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    if training.fine_tune_epochs > 0:
        set_fine_tuning(model, fine_tune_at=training.fine_tune_at)
        compile_model(model, learning_rate=training.fine_tune_learning_rate)
        fine_tune_history = model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=training.fine_tune_epochs,
            callbacks=callbacks,
            verbose=1,
        )
        history_data = _merge_histories(history, fine_tune_history)
    else:
        history_data = _merge_histories(history)

    model_path = project.paths.models_exported / "lettuce_mobilenetv2.keras"
    class_names_path = project.paths.models_exported / "class_names.json"
    history_path = project.paths.reports_metrics / "training_history.json"

    ensure_directory(model_path.parent)
    ensure_directory(history_path.parent)

    model.save(model_path)
    save_json(list(resolved_class_names), class_names_path)
    save_json(history_data, history_path)

    return TrainingArtifacts(
        model_path=model_path,
        history_path=history_path,
        class_names_path=class_names_path,
    )

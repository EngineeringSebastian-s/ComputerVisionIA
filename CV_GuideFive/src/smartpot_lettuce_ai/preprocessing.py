"""TensorFlow preprocessing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .config import DEFAULT_IMAGE_SIZE
from .utils import folderify_label


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "TensorFlow is required for preprocessing, training and inference."
        ) from exc

    return tf


def build_augmentation_model(seed: int = 42):
    """Create a small augmentation stack for image training."""

    tf = _tensorflow()
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(0.08, seed=seed),
            tf.keras.layers.RandomZoom(0.1, seed=seed),
            tf.keras.layers.RandomContrast(0.1, seed=seed),
        ],
        name="augmentation",
    )


def configure_dataset(
    dataset,
    *,
    training: bool,
    shuffle_buffer_size: int = 1000,
    seed: int = 42,
):
    """Apply caching, shuffling and prefetching to a TensorFlow dataset."""

    tf = _tensorflow()
    result = dataset.cache()
    if training:
        result = result.shuffle(shuffle_buffer_size, seed=seed, reshuffle_each_iteration=True)
    return result.prefetch(tf.data.AUTOTUNE)


def create_train_validation_datasets(
    data_dir: Path | str,
    *,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    batch_size: int = 32,
    validation_split: float = 0.2,
    seed: int = 42,
    class_names: Sequence[str] | None = None,
    label_mode: str = "int",
):
    """Build training and validation datasets from a directory tree."""

    tf = _tensorflow()
    directory = Path(data_dir)
    dataset_kwargs = {
        "directory": str(directory),
        "validation_split": validation_split,
        "seed": seed,
        "image_size": image_size,
        "batch_size": batch_size,
        "label_mode": label_mode,
    }
    if class_names is not None:
        dataset_kwargs["class_names"] = [folderify_label(name) for name in class_names]

    train_ds = tf.keras.utils.image_dataset_from_directory(
        subset="training",
        **dataset_kwargs,
    )
    validation_ds = tf.keras.utils.image_dataset_from_directory(
        subset="validation",
        **dataset_kwargs,
    )

    inferred_class_names = (
        [folderify_label(name) for name in class_names]
        if class_names is not None
        else list(train_ds.class_names)
    )
    return (
        configure_dataset(train_ds, training=True, seed=seed),
        configure_dataset(validation_ds, training=False, seed=seed),
        inferred_class_names,
    )


def create_test_dataset(
    data_dir: Path | str,
    *,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    batch_size: int = 32,
    class_names: Sequence[str] | None = None,
    label_mode: str = "int",
):
    """Build an evaluation dataset from a directory tree."""

    tf = _tensorflow()
    directory = Path(data_dir)
    dataset_kwargs = {
        "directory": str(directory),
        "image_size": image_size,
        "batch_size": batch_size,
        "label_mode": label_mode,
        "shuffle": False,
    }
    if class_names is not None:
        dataset_kwargs["class_names"] = [folderify_label(name) for name in class_names]

    dataset = tf.keras.utils.image_dataset_from_directory(**dataset_kwargs)
    inferred_class_names = (
        [folderify_label(name) for name in class_names]
        if class_names is not None
        else list(dataset.class_names)
    )
    return configure_dataset(dataset, training=False), inferred_class_names


def load_image_for_prediction(
    image_path: Path | str,
    *,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> "np.ndarray":
    """Load and resize a single image ready for model inference."""

    import numpy as np
    from PIL import Image

    image = Image.open(image_path).convert("RGB").resize(image_size)
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)

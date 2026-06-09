"""Model factories for the lettuce disease classifier."""

from __future__ import annotations

from typing import Any

from .config import DEFAULT_IMAGE_SIZE


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("TensorFlow is required to build and train the model.") from exc

    return tf


def build_mobilenetv2_classifier(
    *,
    num_classes: int,
    input_shape: tuple[int, int, int] = (*DEFAULT_IMAGE_SIZE, 3),
    dropout_rate: float = 0.2,
    base_trainable: bool = False,
):
    """Build a MobileNetV2 transfer-learning classifier."""

    tf = _tensorflow()
    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescaling")(inputs)
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    base_model.trainable = base_trainable
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="smartpot_lettuce_mobilenetv2")
    return model


def compile_model(model: Any, *, learning_rate: float = 1e-4) -> None:
    """Compile the classifier with an optimizer and loss for sparse labels."""

    tf = _tensorflow()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def set_fine_tuning(model: Any, *, fine_tune_at: int) -> None:
    """Unfreeze the last layers of the nested MobileNetV2 base model."""

    tf = _tensorflow()
    base_model = next(
        (
            layer
            for layer in model.layers
            if isinstance(layer, tf.keras.Model) and layer.name != model.name
        ),
        None,
    )
    if base_model is None:
        raise ValueError("The model does not contain a nested base model to fine-tune.")

    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

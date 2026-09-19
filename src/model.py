"""Multi-task LSTM neural architecture for dual-language sign recognition.

Architecture specified in project ground truth:
    Input: (30, 258)
    -> LSTM(64, return_sequences=True)
    -> LSTM(128, return_sequences=True)
    -> LSTM(64)
    -> Dense(64, relu) -> Dropout(0.3)
    -> Class Head: Dense(num_classes, softmax) [Default: 40 classes]
    -> Language Head: Dense(2, softmax) [ISL vs ASL]

Joint Optimization Objective:
    Total Loss = 1.0 * class_loss + 0.3 * language_loss
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from src.features.normalize import FEATURE_DIM

DEFAULT_SEQUENCE_LENGTH: int = 30
DEFAULT_NUM_CLASSES: int = 40


def build_sign_model(
    num_classes: int = DEFAULT_NUM_CLASSES,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    feature_dim: int = FEATURE_DIM,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Build and compile the dual-head LSTM model.

    Args:
        num_classes: Number of sign classes (e.g. 40 for full 20 concepts x 2 languages).
        sequence_length: Sequence frame length (default: 30).
        feature_dim: Dimension of normalized features per frame (default: 258).
        learning_rate: Adam optimizer initial learning rate.

    Returns:
        Compiled tf.keras.Model with two named outputs: 'class_output' and 'language_output'.
    """
    inputs = layers.Input(shape=(sequence_length, feature_dim), name="landmark_sequence")

    # Three-layer recurrent backbone
    x = layers.LSTM(64, return_sequences=True, name="lstm_layer_1")(inputs)
    x = layers.LSTM(128, return_sequences=True, name="lstm_layer_2")(x)
    x = layers.LSTM(64, return_sequences=False, name="lstm_layer_3")(x)

    # Shared dense representation
    shared_dense = layers.Dense(64, activation="relu", name="dense_representation")(x)
    dropout_feat = layers.Dropout(0.3, name="dropout")(shared_dense)

    # Primary Output: 40-class classification
    class_output = layers.Dense(
        num_classes, activation="softmax", name="class_output"
    )(dropout_feat)

    # Secondary Output: 2-class language discrimination (ISL vs ASL)
    language_output = layers.Dense(
        2, activation="softmax", name="language_output"
    )(dropout_feat)

    model = models.Model(
        inputs=inputs,
        outputs=[class_output, language_output],
        name="isl_asl_translator_model",
    )

    losses = {
        "class_output": "categorical_crossentropy",
        "language_output": "categorical_crossentropy",
    }
    loss_weights = {
        "class_output": 1.0,
        "language_output": 0.3,
    }
    metrics = {
        "class_output": ["accuracy"],
        "language_output": ["accuracy"],
    }

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics,
    )

    return model


def load_inference_model(model_path: str | Path) -> tf.keras.Model:
    """Load a trained Keras model from disk."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {path}")
    return tf.keras.models.load_model(str(path))

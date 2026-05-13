"""
Multi-output CNN for full-CAPTCHA recognition.

Input:  (80, 200, 1)  — full grayscale CAPTCHA image, normalized [0, 1]
Output: 4 × Dense(36, softmax) — one softmax head per character position

Architecture (Functional API — Sequential cannot support multiple outputs):
    Conv2D(32)  → BN → ReLU → MaxPool(2×2)  →  40×100×32
    Conv2D(64)  → BN → ReLU → MaxPool(2×2)  →  20×50×64
    Conv2D(128) → BN → ReLU → MaxPool(2×2)  →  10×25×128
    Flatten (32,000) → Dense(256, relu) → Dropout(0.4)
    char1..char4: Dense(36, softmax)
"""

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Activation,
    MaxPooling2D, Flatten, Dense, Dropout,
)

INPUT_SHAPE = (80, 200, 1)
NUM_CLASSES = 36
N_CHARS = 4


def build_model():
    """Return an uncompiled multi-output Keras model."""
    inputs = Input(shape=INPUT_SHAPE, name="image")

    x = Conv2D(32, (3, 3), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2))(x)          # → 40×100×32

    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2))(x)          # → 20×50×64

    x = Conv2D(128, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2))(x)          # → 10×25×128

    x = Flatten()(x)                     # → 32,000
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)

    outputs = [
        Dense(NUM_CLASSES, activation="softmax", name=f"char{i + 1}")(x)
        for i in range(N_CHARS)
    ]

    return Model(inputs=inputs, outputs=outputs)

"""
CRNN (CNN + BiLSTM) for full-CAPTCHA recognition with CTC loss.

Input:  (32, 200, 1)  — full grayscale CAPTCHA image, normalized [0, 1]
Output: (T, NUM_CLASSES + 1) softmax sequence — per-timestep class probabilities,
        where the extra class (index NUM_CLASSES) is the CTC blank token.

Why CRNN+CTC over the old multi-output CNN:
    The previous model used GlobalAveragePooling, which collapsed all spatial
    information — every per-position head then saw the same vector and could not
    tell character 1 from character 4 (val accuracy stuck near the 1/36 random
    baseline). The CRNN keeps the horizontal axis as a sequence: the CNN extracts
    glyph features column by column, the BiLSTM models left-right context, and
    CTC aligns that sequence to the 4-character label without segmentation.

Architecture (Functional API):
    Conv(64)  -> BN -> ReLU -> MaxPool(2,2)   ->  16x100x64
    Conv(128) -> BN -> ReLU -> MaxPool(2,2)   ->   8x50x128
    Conv(256) -> BN -> ReLU -> MaxPool(2,1)   ->   4x50x256   (pool height only)
    Conv(256) -> BN -> ReLU -> MaxPool(2,1)   ->   2x50x256   (preserve width=timesteps)
    Permute (W,H,C) -> Reshape (T=50, 512)
    Dense(64, relu)
    BiLSTM(128) -> BiLSTM(128)                ->  (50, 256)
    Dense(NUM_CLASSES+1, softmax)             ->  (50, 37)

Time axis = image WIDTH (reading order), enforced by the Permute before reshape.
TIME_STEPS (50) is asserted >= N_CHARS so CTC always has room to align.
"""

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Activation, MaxPooling2D,
    Permute, Reshape, Dense, Bidirectional, LSTM,
)

INPUT_SHAPE = (32, 200, 1)
NUM_CLASSES = 36          # 0-9, A-Z
N_CHARS = 4               # characters per CAPTCHA (label length)
TIME_STEPS = 50           # width after pooling (200 -> 100 -> 50); the CTC time axis
RNN_UNITS = 128

# CTC requires at least one timestep per label character (more, to allow blanks).
assert TIME_STEPS >= N_CHARS, (
    f"TIME_STEPS ({TIME_STEPS}) must be >= N_CHARS ({N_CHARS}) for CTC alignment. "
    f"If you change the width-pooling, update TIME_STEPS to match."
)


def _conv_block(x, filters, pool):
    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool)(x)
    return x


def build_model():
    """Return the uncompiled CRNN. Output is the per-timestep softmax sequence.

    This is the *inference* graph (image -> softmax). train.py wraps it with a
    CTC endpoint layer for training; evaluate.py decodes its output directly.
    """
    inputs = Input(shape=INPUT_SHAPE, name="image")

    x = _conv_block(inputs, 64,  (2, 2))   # -> 16x100x64
    x = _conv_block(x,      128, (2, 2))   # ->  8x50x128
    x = _conv_block(x,      256, (2, 1))   # ->  4x50x256  (height only)
    x = _conv_block(x,      256, (2, 1))   # ->  2x50x256  (height only)

    # Make WIDTH the time axis: (H, W, C) -> (W, H, C) -> (W, H*C)
    x = Permute((2, 1, 3))(x)              # -> (50, 2, 256)
    _, w, h, c = x.shape
    x = Reshape((w, h * c))(x)             # -> (50, 512)

    x = Dense(64, activation="relu")(x)
    x = Bidirectional(LSTM(RNN_UNITS, return_sequences=True))(x)
    x = Bidirectional(LSTM(RNN_UNITS, return_sequences=True))(x)

    # +1 for the CTC blank token (index NUM_CLASSES).
    outputs = Dense(NUM_CLASSES + 1, activation="softmax", name="ctc_softmax")(x)

    model = Model(inputs=inputs, outputs=outputs, name="crnn_ctc")

    # Defensive check: the realized time dimension must match TIME_STEPS.
    realized_t = model.output_shape[1]
    assert realized_t == TIME_STEPS, (
        f"Realized timesteps ({realized_t}) != TIME_STEPS ({TIME_STEPS}). "
        f"Update TIME_STEPS to {realized_t}."
    )
    return model

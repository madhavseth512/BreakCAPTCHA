"""
Training script for the CRNN + CTC CAPTCHA model.

Loads from data_dir:
    X_train.npy, X_val.npy           shape (N, 32, 200, 1)
    y_train.npy, y_val.npy           shape (N, 4)  int32  (character indices)
    char_classes.json

Saves to output_dir:
    captcha_model.h5      — best inference model (image -> softmax), lowest val_loss
    char_classes.json     — copied for evaluate.py / the extension
    training_history.png  — total loss + per-char & full-CAPTCHA accuracy curves

Two graphs, shared weights:
    * inference model : image -> softmax (T, 37)        [saved, used by evaluate.py]
    * training model  : [image, label] -> CTC loss      [optimized here]
The CTCLayer owns the loss via add_loss(), so the training model is compiled
with an optimizer and NO `loss=` argument (the keras.io captcha-OCR pattern).

VALIDATION GATE 4 (do this before any full run):
    python -m model.train --overfit 100
The model MUST reach ~100% full-CAPTCHA accuracy on 100 samples. If it cannot
memorize 100 images, the loss/decode wiring is broken — fix that before training
for real. This catches the failure mode that produced the earlier dead runs.

Usage:
    python -m model.train
    python -m model.train --overfit 100
    python -m model.train --data data/processed --output model/saved_model --epochs 50 --batch-size 32
"""

import os
import sys
import json
import shutil
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
import tensorflow.keras.backend as K

# Support 'python model/train.py' in addition to 'python -m model.train'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model_architecture import build_model, N_CHARS, TIME_STEPS

SEED = 42


class CTCLayer(Layer):
    """Endpoint layer: computes CTC loss from (labels, softmax) and registers it.

    input_length is constant (TIME_STEPS) and label_length is constant (N_CHARS)
    because every synthetic CAPTCHA is exactly N_CHARS characters.
    """

    def call(self, y_true, y_pred):
        batch = tf.shape(y_pred)[0]
        input_length = tf.fill((batch, 1), TIME_STEPS)
        label_length = tf.fill((batch, 1), N_CHARS)
        loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        self.add_loss(tf.reduce_mean(loss))
        return y_pred


def build_training_model(inference_model):
    """Wrap the inference CRNN with a label input + CTC endpoint layer."""
    label_input = Input(shape=(N_CHARS,), dtype="int32", name="label")
    ctc_out = CTCLayer(name="ctc_loss")(label_input, inference_model.output)
    return Model(inputs=[inference_model.input, label_input], outputs=ctc_out)


def ctc_greedy_decode(inference_model, X, batch_size=64):
    """Run inference and greedy-CTC-decode to integer sequences (N, <=N_CHARS)."""
    softmax = inference_model.predict(X, batch_size=batch_size, verbose=0)
    input_length = np.full((softmax.shape[0],), TIME_STEPS)
    decoded, _ = K.ctc_decode(softmax, input_length=input_length, greedy=True)
    seq = K.get_value(decoded[0])  # (N, max_len), padded with -1
    return seq


def sequence_accuracy(decoded, y_true):
    """Per-character and full-sequence accuracy against fixed-length y_true (N,4)."""
    n = len(y_true)
    # Normalize decoded width to N_CHARS (pad short, truncate long) with -1.
    fixed = np.full((n, N_CHARS), -1, dtype="int64")
    width = min(decoded.shape[1], N_CHARS)
    fixed[:, :width] = decoded[:, :width]

    char_correct = (fixed == y_true).sum()
    char_total = n * N_CHARS
    full_correct = np.all(fixed == y_true, axis=1).sum()
    return char_correct / char_total, full_correct / n


class DecodeAccuracy(Callback):
    """After each epoch, greedy-decode val set and log real accuracy.

    CTC loss alone is hard to read; this restores per-char / full-CAPTCHA
    monitoring. Values are written into `logs` so they land in History and
    can be plotted afterwards.
    """

    def __init__(self, inference_model, X_val, y_val):
        super().__init__()
        self.inference_model = inference_model
        self.X_val = X_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        decoded = ctc_greedy_decode(self.inference_model, self.X_val)
        char_acc, full_acc = sequence_accuracy(decoded, self.y_val)
        logs["val_char_acc"] = char_acc
        logs["val_full_acc"] = full_acc
        print(f"  val_char_acc: {char_acc:.4f}  |  val_full_acc: {full_acc:.4f}")


def save_history_plot(history, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["loss"], label="train")
    if "val_loss" in history.history:
        axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("CTC Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    if "val_char_acc" in history.history:
        axes[1].plot(history.history["val_char_acc"], label="per-char")
    if "val_full_acc" in history.history:
        axes[1].plot(history.history["val_full_acc"], label="full CAPTCHA")
    axes[1].set_title("Validation Accuracy (CTC-decoded)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "training_history.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"Training history plot saved -> '{path}'")


def load_data(data_dir):
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_val   = np.load(os.path.join(data_dir, "X_val.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_val   = np.load(os.path.join(data_dir, "y_val.npy"))
    return X_train, X_val, y_train, y_val


def make_dataset(X, y, batch_size, shuffle):
    """Inputs-only dataset: the CTC loss is internal, so no separate targets.

    Training batches drop the final remainder. A tiny last batch (e.g. 4 samples)
    produces garbage BatchNorm statistics that poison the moving averages used at
    inference time — which silently breaks decoding even when the training loss
    looks healthy. Validation keeps every sample (BN is not updated during eval).
    """
    ds = tf.data.Dataset.from_tensor_slices({"image": X, "label": y})
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(X), 4096), seed=SEED)
    return ds.batch(batch_size, drop_remainder=shuffle).prefetch(tf.data.AUTOTUNE)


def run_overfit_check(data_dir, n, batch_size):
    """Gate 4: confirm the pipeline can memorize a tiny subset (~100% full acc)."""
    print(f"\n=== OVERFIT GATE: memorize {n} samples (must reach ~100% full acc) ===\n")
    X_train, _, y_train, _ = load_data(data_dir)
    X, y = X_train[:n], y_train[:n]

    inference_model = build_model()
    training_model = build_training_model(inference_model)
    # lr=2e-3 descends cleanly under mini-batch noise (5e-3 bounces); the dropped
    # remainder (see make_dataset) keeps BatchNorm inference stats clean so the
    # decode actually reflects what the model learned.
    training_model.compile(optimizer=Adam(2e-3))

    ds = make_dataset(X, y, batch_size, shuffle=True)
    acc_cb = DecodeAccuracy(inference_model, X, y)  # train==val here on purpose
    training_model.fit(ds, epochs=300, callbacks=[acc_cb], verbose=2)

    decoded = ctc_greedy_decode(inference_model, X)
    char_acc, full_acc = sequence_accuracy(decoded, y)
    print(f"\nFinal overfit accuracy -- per-char: {char_acc:.4f}  |  full: {full_acc:.4f}")
    if full_acc >= 0.95:
        print("GATE 4 PASSED [OK]  Wiring is correct -- safe to launch a full run.")
    else:
        print("GATE 4 FAILED [X]  Model cannot memorize 100 samples -- DO NOT train for real.")
        print("Check: label encoding, blank-token index, time-axis permute, decode width.")


def train(data_dir, output_dir, epochs, batch_size):
    os.makedirs(output_dir, exist_ok=True)
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs detected: {gpus if gpus else 'none — training on CPU'}")

    print("\nLoading preprocessed data...")
    X_train, X_val, y_train, y_val = load_data(data_dir)
    print(f"  X_train: {X_train.shape}  |  X_val: {X_val.shape}")
    print(f"  y_train: {y_train.shape}  |  y_val: {y_val.shape}")

    src_classes = os.path.join(data_dir, "char_classes.json")
    dst_classes = os.path.join(output_dir, "char_classes.json")
    shutil.copy(src_classes, dst_classes)
    with open(dst_classes) as f:
        classes = json.load(f)
    print(f"\nClasses ({len(classes)}): {classes}")

    inference_model = build_model()
    training_model = build_training_model(inference_model)
    # lr=2e-3 descends cleanly under mini-batch gradient noise (5e-3 bounces);
    # clipnorm guards against the loss spikes that blew up an earlier run, and
    # ReduceLROnPlateau anneals once progress stalls.
    training_model.compile(optimizer=Adam(learning_rate=2e-3, clipnorm=1.0))
    inference_model.summary()

    model_path = os.path.join(output_dir, "captcha_model.h5")
    callbacks = [
        DecodeAccuracy(inference_model, X_val, y_val),
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1),
        # Save the INFERENCE model (image -> softmax); that is what evaluate.py loads.
        ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, verbose=1),
    ]

    train_ds = make_dataset(X_train, y_train, batch_size, shuffle=True)
    val_ds   = make_dataset(X_val,   y_val,   batch_size, shuffle=False)

    print(f"\nTraining for up to {epochs} epochs  (batch_size={batch_size})...\n")
    history = training_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # ModelCheckpoint saved the training graph; re-save the clean inference model
    # at the best (restored) weights so evaluate.py / TF.js get image -> softmax.
    inference_model.save(model_path)
    print(f"Inference model saved -> '{model_path}'")

    if "val_full_acc" in history.history:
        best = max(history.history["val_full_acc"])
        print(f"\nBest val full-CAPTCHA accuracy: {best:.4f}  ({best*100:.2f}%)")

    save_history_plot(history, output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CRNN+CTC CAPTCHA model.")
    parser.add_argument("--data",       type=str, default="data/processed")
    parser.add_argument("--output",     type=str, default="model/saved_model")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument("--overfit",    type=int, default=0,
                        help="Gate 4: train+eval on this many samples to verify wiring (e.g. 100)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.overfit > 0:
        run_overfit_check(args.data, args.overfit, args.batch_size)
    else:
        train(
            data_dir=args.data,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

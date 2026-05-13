"""
Training script for the multi-output CAPTCHA CNN.

Loads from data_dir:
    X_train.npy, X_val.npy           shape (N, 80, 200, 1)
    y1_train.npy .. y4_train.npy     shape (N, 36)  one-hot per character position
    y1_val.npy   .. y4_val.npy
    char_classes.json

Saves to output_dir:
    captcha_model.h5      — best checkpoint (lowest val_loss)
    char_classes.json     — copied from data_dir for use by evaluate.py and the extension
    training_history.png  — total loss + per-head accuracy curves

Usage:
    python model/train.py
    python model/train.py --data data/processed --output model/saved_model --epochs 30 --batch-size 32
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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Support 'python model/train.py' from project root in addition to 'python -m model.train'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model_architecture import build_model, N_CHARS


def save_history_plot(history, output_dir):
    heads = [f"char{i + 1}" for i in range(N_CHARS)]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axs = axes.flat

    ax = next(axs)
    ax.plot(history.history["loss"], label="train")
    ax.plot(history.history["val_loss"], label="val")
    ax.set_title("Total Loss")
    ax.set_xlabel("Epoch")
    ax.legend()

    for head in heads:
        ax = next(axs)
        ax.plot(history.history[f"{head}_accuracy"], label="train")
        ax.plot(history.history[f"val_{head}_accuracy"], label="val")
        ax.set_title(f"{head} Accuracy")
        ax.set_xlabel("Epoch")
        ax.legend()

    next(axs).set_visible(False)  # 6th cell unused

    plt.tight_layout()
    path = os.path.join(output_dir, "training_history.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"Training history plot saved → '{path}'")


def train(data_dir, output_dir, epochs, batch_size):
    os.makedirs(output_dir, exist_ok=True)

    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs detected: {gpus if gpus else 'none — training on CPU'}")

    # --- Load data ---
    print("\nLoading preprocessed data...")
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_val   = np.load(os.path.join(data_dir, "X_val.npy"))
    y_train = {f"char{i+1}": np.load(os.path.join(data_dir, f"y{i+1}_train.npy")) for i in range(N_CHARS)}
    y_val   = {f"char{i+1}": np.load(os.path.join(data_dir, f"y{i+1}_val.npy"))   for i in range(N_CHARS)}

    print(f"  X_train: {X_train.shape}  |  X_val: {X_val.shape}")
    for k, v in y_train.items():
        print(f"  {k}_train: {v.shape}")

    # --- Copy char_classes.json to output_dir ---
    src_classes = os.path.join(data_dir, "char_classes.json")
    dst_classes = os.path.join(output_dir, "char_classes.json")
    shutil.copy(src_classes, dst_classes)
    with open(dst_classes) as f:
        classes = json.load(f)
    print(f"\nClasses ({len(classes)}): {classes}")

    # --- Build & compile ---
    model = build_model()
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={f"char{i+1}": "categorical_crossentropy" for i in range(N_CHARS)},
        metrics={f"char{i+1}": "accuracy" for i in range(N_CHARS)},
    )
    model.summary()

    # --- Callbacks ---
    model_path = os.path.join(output_dir, "captcha_model.h5")
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=5,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=3, verbose=1,
        ),
        ModelCheckpoint(
            model_path, monitor="val_loss",
            save_best_only=True, verbose=1,
        ),
    ]

    # --- Train ---
    print(f"\nTraining for up to {epochs} epochs  (batch_size={batch_size})...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    # --- Report at best epoch ---
    best_epoch = int(np.argmin(history.history["val_loss"]))
    print(f"\n--- Results at best epoch ({best_epoch + 1}) ---")
    per_head_accs = []
    for i in range(N_CHARS):
        acc = history.history[f"val_char{i+1}_accuracy"][best_epoch]
        per_head_accs.append(acc)
        print(f"  char{i+1} val accuracy: {acc:.4f}  ({acc*100:.2f}%)")

    captcha_acc = float(np.prod(per_head_accs))
    print(f"\nEstimated full-CAPTCHA accuracy (product of heads): {captcha_acc:.4f}  ({captcha_acc*100:.2f}%)")
    print(f"Model saved → '{model_path}'")

    save_history_plot(history, output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Train multi-output CAPTCHA CNN.")
    parser.add_argument("--data",       type=str, default="data/processed")
    parser.add_argument("--output",     type=str, default="model/saved_model")
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_dir=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

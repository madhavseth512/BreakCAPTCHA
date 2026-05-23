"""
Evaluation script for the CRNN + CTC CAPTCHA model.

Loads the saved inference model (image -> softmax sequence), greedy-CTC-decodes
the validation set, and reports:
    - Per-position accuracy (char1 .. char4)
    - Full CAPTCHA accuracy (all 4 chars correct)
    - Per-class classification report (sklearn, all positions combined)
    - Top-10 most confused character pairs
    - 20 random sample predictions (console + prediction_samples.png)

Usage:
    python -m model.evaluate
    python -m model.evaluate --model model/saved_model/captcha_model.h5 --data data/processed
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import classification_report, confusion_matrix

N_CHARS = 4


def greedy_decode(model, X, batch_size=64):
    """Inference + greedy CTC decode -> integer sequences padded to N_CHARS with -1."""
    softmax = model.predict(X, batch_size=batch_size, verbose=1)
    time_steps = softmax.shape[1]
    input_length = np.full((softmax.shape[0],), time_steps)
    decoded, _ = K.ctc_decode(softmax, input_length=input_length, greedy=True)
    seq = K.get_value(decoded[0])  # (N, max_len), -1 padded

    fixed = np.full((len(X), N_CHARS), -1, dtype="int64")
    width = min(seq.shape[1], N_CHARS)
    fixed[:, :width] = seq[:, :width]
    return fixed


def evaluate(model_path, data_dir):
    print(f"Loading model from '{model_path}'...")
    model = tf.keras.models.load_model(model_path)

    classes_path = os.path.join(os.path.dirname(model_path), "char_classes.json")
    with open(classes_path) as f:
        classes = json.load(f)
    print(f"Classes ({len(classes)}): {classes}")

    print("\nLoading validation data...")
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))  # (N, 4) int
    print(f"  X_val: {X_val.shape}")

    print("\nRunning inference + CTC decode...")
    pred = greedy_decode(model, X_val)  # (N, 4), -1 = no/short prediction

    # --- Per-position accuracy ---
    print("\n--- Per-position accuracy ---")
    for i in range(N_CHARS):
        acc = np.mean(pred[:, i] == y_val[:, i])
        print(f"  char{i+1}: {acc:.4f}  ({acc*100:.2f}%)")

    # --- Full CAPTCHA accuracy ---
    all_correct = np.all(pred == y_val, axis=1)
    captcha_acc = all_correct.mean()
    print(f"\nFull CAPTCHA accuracy: {captcha_acc:.4f}  ({captcha_acc*100:.2f}%)")
    print(f"  ({all_correct.sum()}/{len(all_correct)} fully correct)")

    # --- Per-class report (positions combined; drop -1 'no prediction' slots) ---
    true_flat = y_val.reshape(-1)
    pred_flat = pred.reshape(-1)
    valid = pred_flat >= 0
    dropped = (~valid).sum()
    if dropped:
        print(f"\nNote: {dropped} position(s) decoded to fewer than {N_CHARS} chars "
              f"(counted as wrong above, excluded from the per-class report).")
    print("\n--- Per-class classification report (all positions combined) ---")
    print(classification_report(
        true_flat[valid], pred_flat[valid],
        labels=list(range(len(classes))), target_names=classes, zero_division=0,
    ))

    # --- Top-10 most confused character pairs ---
    cm = confusion_matrix(true_flat[valid], pred_flat[valid], labels=list(range(len(classes))))
    np.fill_diagonal(cm, 0)
    top_idx = np.argsort(cm.flatten())[::-1][:10]
    print("--- Top-10 most confused character pairs ---")
    for idx in top_idx:
        r, c = divmod(idx, len(classes))
        if cm[r, c] > 0:
            print(f"  True '{classes[r]}' -> Predicted '{classes[c]}': {cm[r, c]} times")

    # --- 20 random sample predictions ---
    rng = np.random.default_rng(seed=0)
    sample_idx = rng.choice(len(X_val), size=min(20, len(X_val)), replace=False)
    print("\n--- 20 random sample predictions ---")
    print(f"  {'Actual':<8} {'Predicted':<10} Result")
    print("  " + "-" * 30)
    for idx in sample_idx:
        actual    = _to_text(y_val[idx], classes)
        predicted = _to_text(pred[idx], classes)
        marker    = "OK" if actual == predicted else "X"
        print(f"  {actual:<8} {predicted:<10} {marker}")

    _save_prediction_samples(X_val, y_val, pred, classes, sample_idx, os.path.dirname(model_path))


def _to_text(seq, classes):
    return "".join(classes[i] if 0 <= i < len(classes) else "_" for i in seq)


def _save_prediction_samples(X_val, y_val, pred, classes, sample_idx, output_dir):
    fig, axes = plt.subplots(4, 5, figsize=(20, 12))
    for ax, idx in zip(axes.flat, sample_idx):
        ax.imshow(X_val[idx].squeeze(), cmap="gray")
        actual    = _to_text(y_val[idx], classes)
        predicted = _to_text(pred[idx], classes)
        color = "green" if actual == predicted else "red"
        ax.set_title(f"A: {actual}\nP: {predicted}", color=color, fontsize=9)
        ax.axis("off")
    plt.suptitle("Actual (A) vs Predicted (P) — green = correct, red = wrong", fontsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, "prediction_samples.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"\nPrediction samples saved -> '{path}'")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CRNN+CTC CAPTCHA model.")
    parser.add_argument("--model", type=str, default="model/saved_model/captcha_model.h5")
    parser.add_argument("--data",  type=str, default="data/processed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.model, args.data)

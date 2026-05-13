"""
Evaluation script for the multi-output CAPTCHA CNN.

Reports:
    - Per-position accuracy (char1 .. char4)
    - Full CAPTCHA accuracy (all 4 chars correct)
    - Per-class classification report (sklearn, all positions combined)
    - Top-10 most confused character pairs
    - 20 random sample predictions (printed to console + saved as prediction_samples.png)

Usage:
    python model/evaluate.py
    python model/evaluate.py --model model/saved_model/captcha_model.h5 --data data/processed
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

N_CHARS = 4


def evaluate(model_path, data_dir):
    print(f"Loading model from '{model_path}'...")
    model = tf.keras.models.load_model(model_path)

    # Load class mapping — never hardcoded, always from char_classes.json
    classes_path = os.path.join(os.path.dirname(model_path), "char_classes.json")
    with open(classes_path) as f:
        classes = json.load(f)
    print(f"Classes ({len(classes)}): {classes}")

    print("\nLoading validation data...")
    X_val  = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val  = [np.load(os.path.join(data_dir, f"y{i+1}_val.npy")) for i in range(N_CHARS)]
    print(f"  X_val: {X_val.shape}")

    print("\nRunning inference...")
    preds        = model.predict(X_val, batch_size=64, verbose=1)
    pred_indices = [np.argmax(p, axis=1) for p in preds]
    true_indices = [np.argmax(y, axis=1) for y in y_val]

    # --- Per-position accuracy ---
    print("\n--- Per-position accuracy ---")
    for i in range(N_CHARS):
        acc = np.mean(pred_indices[i] == true_indices[i])
        print(f"  char{i+1}: {acc:.4f}  ({acc*100:.2f}%)")

    # --- Full CAPTCHA accuracy ---
    all_correct = np.all(
        np.stack([pred_indices[i] == true_indices[i] for i in range(N_CHARS)], axis=1),
        axis=1,
    )
    captcha_acc = all_correct.mean()
    print(f"\nFull CAPTCHA accuracy: {captcha_acc:.4f}  ({captcha_acc*100:.2f}%)")
    print(f"  ({all_correct.sum()}/{len(all_correct)} fully correct)")

    # --- Per-class classification report (all positions combined) ---
    all_true = np.concatenate(true_indices)
    all_pred = np.concatenate(pred_indices)
    print("\n--- Per-class classification report (all positions combined) ---")
    print(classification_report(all_true, all_pred, target_names=classes))

    # --- Top-10 most confused character pairs ---
    cm = confusion_matrix(all_true, all_pred)
    np.fill_diagonal(cm, 0)
    top_idx = np.argsort(cm.flatten())[::-1][:10]
    print("--- Top-10 most confused character pairs ---")
    for idx in top_idx:
        r, c = divmod(idx, len(classes))
        if cm[r, c] > 0:
            print(f"  True '{classes[r]}' → Predicted '{classes[c]}': {cm[r, c]} times")

    # --- 20 random sample predictions ---
    rng = np.random.default_rng(seed=0)
    sample_idx = rng.choice(len(X_val), size=20, replace=False)
    print("\n--- 20 random sample predictions ---")
    print(f"  {'Actual':<8} {'Predicted':<10} Result")
    print("  " + "-" * 30)
    for idx in sample_idx:
        actual    = "".join(classes[true_indices[i][idx]] for i in range(N_CHARS))
        predicted = "".join(classes[pred_indices[i][idx]] for i in range(N_CHARS))
        marker    = "✓" if actual == predicted else "✗"
        print(f"  {actual:<8} {predicted:<10} {marker}")

    output_dir = os.path.dirname(model_path)
    _save_prediction_samples(X_val, true_indices, pred_indices, classes, sample_idx, output_dir)


def _save_prediction_samples(X_val, true_indices, pred_indices, classes, sample_idx, output_dir):
    fig, axes = plt.subplots(4, 5, figsize=(20, 12))
    for ax, idx in zip(axes.flat, sample_idx):
        ax.imshow(X_val[idx].squeeze(), cmap="gray")
        actual    = "".join(classes[true_indices[i][idx]] for i in range(N_CHARS))
        predicted = "".join(classes[pred_indices[i][idx]] for i in range(N_CHARS))
        color = "green" if actual == predicted else "red"
        ax.set_title(f"A: {actual}\nP: {predicted}", color=color, fontsize=9)
        ax.axis("off")
    plt.suptitle("Actual (A) vs Predicted (P) — green = correct, red = wrong", fontsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, "prediction_samples.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"\nPrediction samples saved → '{path}'")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate multi-output CAPTCHA CNN.")
    parser.add_argument("--model", type=str, default="model/saved_model/captcha_model.h5")
    parser.add_argument("--data",  type=str, default="data/processed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.model, args.data)

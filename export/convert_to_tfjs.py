"""
Phase 5: Convert captcha_model.h5 to TensorFlow.js LayersModel.

Prerequisites:
    pip install tensorflowjs==4.10.0
    (If that fails due to jax/jaxlib deps: pip install tensorflowjs==4.10.0 --no-deps)

Usage:
    python -m export.convert_to_tfjs
    python -m export.convert_to_tfjs --model model/saved_model/captcha_model.h5
"""

import os
import sys
import shutil
import argparse


def convert(model_path, tfjs_output, extension_dir):
    try:
        import tensorflowjs as tfjs
    except ImportError:
        print("ERROR: tensorflowjs not installed.")
        print("  Run: pip install tensorflowjs==4.10.0")
        print("  If jax/jaxlib cause failures: pip install tensorflowjs==4.10.0 --no-deps")
        sys.exit(1)

    import tensorflow as tf

    print(f"Loading '{model_path}'...")
    # compile=False avoids loading the custom CTCLayer used during training.
    # The saved model is the inference graph (image -> softmax) so no CTC needed.
    model = tf.keras.models.load_model(model_path, compile=False)
    model.summary()

    os.makedirs(tfjs_output, exist_ok=True)
    print(f"\nConverting to TF.js LayersModel -> '{tfjs_output}'...")
    tfjs.converters.save_keras_model(model, tfjs_output)
    print(f"TF.js model written to '{tfjs_output}/'")

    # Copy char_classes.json so the extension can read the label mapping at runtime.
    model_dir = os.path.dirname(model_path)
    classes_src = os.path.join(model_dir, "char_classes.json")
    shutil.copy(classes_src, os.path.join(tfjs_output, "char_classes.json"))
    print(f"Copied char_classes.json -> '{tfjs_output}/'")

    # Mirror into extension/tfjs_model/ so the extension is ready to load unpacked.
    ext_model_dir = os.path.join(extension_dir, "tfjs_model")
    if os.path.exists(ext_model_dir):
        shutil.rmtree(ext_model_dir)
    shutil.copytree(tfjs_output, ext_model_dir)
    print(f"Mirrored model -> '{ext_model_dir}/'")

    print("\nPhase 5 complete. Next steps:")
    print("  1. Download tf.min.js and place it at extension/lib/tf.min.js")
    print("     https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/dist/tf.min.js")
    print("  2. python extension/generate_icons.py")
    print("  3. chrome://extensions -> Developer mode -> Load unpacked -> extension/")


def parse_args():
    p = argparse.ArgumentParser(description="Export Keras model to TF.js LayersModel.")
    p.add_argument("--model",    type=str, default="model/saved_model/captcha_model.h5")
    p.add_argument("--tfjs-out", type=str, default="export/tfjs_model", dest="tfjs_output")
    p.add_argument("--ext-dir",  type=str, default="extension",          dest="extension_dir")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args.model, args.tfjs_output, args.extension_dir)

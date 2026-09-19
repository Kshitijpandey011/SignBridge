"""TensorFlow Lite export and numerical consistency validation.

Converts Keras .keras multi-output model into an optimized .tflite flatbuffer and validates
that inference outputs between Keras and TFLite match within tight floating-point tolerances.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import load_inference_model


def convert_and_validate(
    keras_model_path: str = "models/model.keras",
    tflite_output_path: str = "models/model.tflite",
    tolerance_atol: float = 1e-3,
) -> bool:
    """Convert Keras model to TFLite and verify numerical equivalence on synthetic test data."""
    print(f"Loading Keras model from: {keras_model_path}")
    model = load_inference_model(keras_model_path)

    # 1. Convert to TFLite
    print("Converting model to TensorFlow Lite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Enable select TF ops for LSTM recurrent cells
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False

    tflite_model = converter.convert()

    out_file = Path(tflite_output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        f.write(tflite_model)
    print(f"Saved TFLite model to: {out_file} ({len(tflite_model) / 1024:.1f} KB)")

    # 2. Verify with TFLite Interpreter
    print("\nValidating numerical parity between Keras and TFLite...")
    interpreter = tf.lite.Interpreter(model_path=str(out_file))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Generate synthetic validation batch
    rng = np.random.RandomState(42)
    test_batch = rng.normal(0.0, 0.5, size=(5, 30, 258)).astype(np.float32)

    keras_class_preds, keras_lang_preds = model.predict(test_batch, verbose=0)

    # Run TFLite inference per sample
    tflite_class_preds = []
    tflite_lang_preds = []

    # Map output indices
    class_out_idx = next(
        d["index"] for d in output_details if "class" in d["name"].lower() or d["shape"][-1] > 2
    )
    lang_out_idx = next(
        d["index"] for d in output_details if "lang" in d["name"].lower() or d["shape"][-1] == 2
    )

    for i in range(len(test_batch)):
        sample = np.expand_dims(test_batch[i], axis=0)
        interpreter.set_tensor(input_details[0]["index"], sample)
        interpreter.invoke()

        pred_class = interpreter.get_tensor(class_out_idx)
        pred_lang = interpreter.get_tensor(lang_out_idx)

        tflite_class_preds.append(pred_class[0])
        tflite_lang_preds.append(pred_lang[0])

    tflite_class_preds = np.array(tflite_class_preds)
    tflite_lang_preds = np.array(tflite_lang_preds)

    max_diff_class = float(np.max(np.abs(keras_class_preds - tflite_class_preds)))
    max_diff_lang = float(np.max(np.abs(keras_lang_preds - tflite_lang_preds)))

    print(f"Max Absolute Error (Class Head)   : {max_diff_class:.6e}")
    print(f"Max Absolute Error (Language Head): {max_diff_lang:.6e}")

    passed = max_diff_class <= tolerance_atol and max_diff_lang <= tolerance_atol
    if passed:
        print("[SUCCESS] TFLite model output matches Keras predictions within tolerance!")
    else:
        print(f"[WARNING] Discrepancy exceeded atol={tolerance_atol}")

    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Keras model to TFLite")
    parser.add_argument("--model", type=str, default="models/model.keras", help="Path to input Keras model")
    parser.add_argument("--output", type=str, default="models/model.tflite", help="Path for output TFLite file")
    args = parser.parse_args()

    convert_and_validate(keras_model_path=args.model, tflite_output_path=args.output)


if __name__ == "__main__":
    main()

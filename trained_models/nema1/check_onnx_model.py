#!/usr/bin/env python3
"""Check ONNX model inputs and outputs."""

import onnx
import onnxruntime as ort

def check_model(model_path):
    print(f"\nChecking {model_path}:")
    print("-" * 50)

    # Load the model
    model = onnx.load(model_path)

    # Check inputs
    print("Inputs:")
    for input in model.graph.input:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in input.type.tensor_type.shape.dim]
        print(f"  - {input.name}: {shape} ({onnx.TensorProto.DataType.Name(input.type.tensor_type.elem_type)})")

    # Check outputs
    print("\nOutputs:")
    for output in model.graph.output:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in output.type.tensor_type.shape.dim]
        print(f"  - {output.name}: {shape} ({onnx.TensorProto.DataType.Name(output.type.tensor_type.elem_type)})")

    # Try to create session to verify
    try:
        session = ort.InferenceSession(model_path)
        print(f"\n✅ Model can be loaded by ONNX Runtime")
        print(f"Input names: {session.get_inputs()[0].name}")
        print(f"Output names: {session.get_outputs()[0].name}")
    except Exception as e:
        print(f"\n❌ Error loading with ONNX Runtime: {e}")

if __name__ == "__main__":
    # Check both models
    check_model("onnx_rare_words_epoch80/model_fp32.onnx")
    check_model("onnx_rare_words_epoch80/encoder.onnx")
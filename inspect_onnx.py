
import onnx

encoder_path = '/home/will/git/swype/cleverkeys/web-demo/models/rnnt_new/encoder.onnx'
decoder_joint_path = '/home/will/git/swype/cleverkeys/web-demo/models/rnnt_new/decoder_joint.onnx'

print("--- Encoder Info ---")
encoder_model = onnx.load(encoder_path)
print("Inputs:", [inp.name for inp in encoder_model.graph.input])
print("Outputs:", [out.name for out in encoder_model.graph.output])

print("\n--- Decoder-Joint Info ---")
decoder_joint_model = onnx.load(decoder_joint_path)
print("Inputs:", [inp.name for inp in decoder_joint_model.graph.input])
print("Outputs:", [out.name for out in decoder_joint_model.graph.output])

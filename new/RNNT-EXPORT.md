## How Stateful RNN-T Export Works
Your script creates two models, which is the correct approach for efficient, stateful RNN-T inference:

encoder.onnx: This model is stateless. It processes chunks of input features and produces acoustic encodings. It's run once per chunk.

decoder_joint.onnx: This model is stateful. It contains the RNN-T Prediction Network (the LSTM decoder) and the Joint Network. It must be run in a loop, one acoustic timestep at a time, and its internal state must be manually managed between calls.

The "struggle with the decoder state" comes from managing the hidden state of the Prediction Network's internal LSTM.

According to the NeMo documentation, the RNNTDecoder's predict method, which is what gets exported to ONNX, has the following signature:


Inputs: It takes the previously predicted token (y) and an optional previous state . For an LSTM, this 



state is a tuple of two tensors: the hidden state (h) and the cell state (c).


Outputs: It returns the prediction network output (g) and the new, updated state (hid).

This means your decoder_joint.onnx model expects the previous state as an input and provides the next state as an output. You must capture this output state and feed it back as an input in the next iteration of your decoding loop.

## What Your Script is Doing Correctly
Your export_stateful_pair.py script is correctly preparing for this state management by creating the runtime_meta.json file. Specifically, it extracts two crucial pieces of information:


num_layers: The number of LSTM layers in the prediction network.



hidden_size: The hidden dimension of the LSTM.


Your inference code on the Android device must use these values to initialize the very first state tensors. The initial state is typically a pair of zero-tensors, each with the shape 

[num_layers, batch_size, hidden_size] (where batch_size is 1 for single-swipe decoding) .

## Likely Point of Failure: The Inference Loop
The problem is almost certainly in your inference code that runs the ONNX models. A correct stateful inference loop should follow this logic:

Python

# Pseudo-code for a correct stateful RNN-T inference loop

# 1. Load ONNX models and runtime_meta.json
encoder = onnx.load("encoder.onnx")
decoder_joint = onnx.load("decoder_joint.onnx")
meta = json.load("runtime_meta.json")

# 2. Preprocess swipe into a feature tensor `features`
# ...

# 3. Run the stateless encoder ONCE
encoder_output = encoder.run(features)  # Shape: [1, T, D_encoder]

# 4. Initialize the first decoder state using metadata
num_layers = meta['decoder_config']['num_layers']
hidden_size = meta['decoder_config']['hidden_size']
batch_size = 1
h_0 = np.zeros((num_layers, batch_size, hidden_size), dtype=np.float32)
c_0 = np.zeros((num_layers, batch_size, hidden_size), dtype=np.float32)

# 5. Initialize the first input token (Start-of-Sequence) and hypothesis
# The predictor's BOS token is typically the blank token ID.
prev_token = np.array([[meta['blank_id']]], dtype=np.int64)
hypothesis = []
current_state_h, current_state_c = h_0, c_0

# 6. Loop over each acoustic timestep from the encoder output
for t in range(encoder_output.shape[1]):
    acoustic_frame = encoder_output[:, t:t+1, :] # Current acoustic frame

    # Loop to consume multiple non-blank tokens for a single acoustic frame
    while True:
        # 7. Run the stateful decoder_joint model
        # NOTE: Input/output names MUST match the ONNX file. Use Netron to verify.
        onnx_inputs = {
            'encoder_output': acoustic_frame,
            'prev_token': prev_token,
            'prev_state_h': current_state_h,
            'prev_state_c': current_state_c
        }
        
        # The model MUST output the new state
        logits, new_state_h, new_state_c = decoder_joint.run(onnx_inputs)

        # 8. Get the most likely token and UPDATE the state for the NEXT iteration
        predicted_token_id = np.argmax(logits, axis=-1)
        current_state_h, current_state_c = new_state_h, new_state_c
        
        # 9. Check for blank and update hypothesis
        if predicted_token_id == meta['blank_id']:
            break  # Move to the next acoustic frame `t`
        else:
            hypothesis.append(predicted_token_id.item())
            prev_token = predicted_token_id # This predicted token becomes the next input
## How to Debug
Inspect the ONNX Models with Netron. This is the most important step. Open decoder_joint.onnx in a tool like Netron. This will show you the exact input and output names and their expected shapes. Your ONNX runtime code must use these exact names when providing inputs and retrieving outputs. The names will likely be something like prev_state_h, prev_state_c for inputs and new_state_h, new_state_c for outputs.

Verify State Tensor Shapes. In your inference code, print the shapes of the state tensors you are creating and passing to the model. Ensure they match the shape (num_layers, 1, hidden_size) derived from runtime_meta.json and confirmed in Netron.

Confirm State Passthrough. Ensure your loop correctly takes the new state tensors from the model's output and uses them as the input for the very next call to the model. Any mistake here will break the auto-regressive decoding process.
Of course. The graph you've shared is a perfect visual representation of the stateful decoding process we discussed, and it confirms your export script is working correctly. The key to solving your problem lies in understanding how to feed this graph the state it needs and how to retrieve the new state it outputs.

Let's walk through the graph you provided.

## A Guided Tour of Your ONNX Graph
This graph shows exactly how the decoder_joint.onnx model works for a single step of decoding.

1. The State Inputs (Your Responsibility) 🧠

At the very top, you see input_states_1 and input_states_2.

These are the hidden state (h) and cell state (c) of the decoder's internal LSTM.

Your inference code must provide these two tensors every time you run the model. For the very first step, you'll create them as zero-tensors using the num_layers and hidden_size from your runtime_meta.json file.

2. The LSTM Core ➡️

The /decoder/d.../LSTM node is the heart of the prediction network.

It takes three things: the previous token (which comes from the targets input after an embedding lookup) and your two input_states.

Its job is to predict the next step based on what it has seen before.

3. The State Outputs (The Crucial Part) ✨

The LSTM node produces three outputs. Two of them are labeled output_states_1 and output_states_2.

These are the new, updated hidden and cell states for the next timestep.

Your code must capture these two outputs. After running the model for one step, you save these two tensors and use them as input_states_1 and input_states_2 for the very next step.

4. The Joint Network and Final Output

The third output from the LSTM goes down to the right side of the joint network (/joint/pred/MatMul).

The encoder_outputs (from the other ONNX model) come in on the left.

The graph then combines them, adds them, and passes them through a few more layers (Relu, MatMul) to produce the final outputs at the bottom. These are the logits you use to find the most likely next token.

## Summary of the Inference Flow 🔄
Based on this graph, your code's loop for each acoustic frame must do the following:

Provide Inputs: Feed the model the current encoder_outputs, the last predicted targets token, and the current state tensors (input_states_1 and input_states_2).

Receive Outputs: Get the final outputs (logits) from the bottom of the graph, but also get the output_states_1 and output_states_2 from the middle.

Update and Repeat: Use the new output_states as the input_states for the next iteration of the loop.

Your struggle is almost certainly happening because your code is either not providing the input states or not capturing and reusing the output states. By matching your inference logic to the inputs and outputs shown in this graph, you should be able to get it working.
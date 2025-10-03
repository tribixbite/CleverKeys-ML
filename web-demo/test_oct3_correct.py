import numpy as np
import onnxruntime as ort
import json
import sys
sys.path.insert(0, '../new')
from train_transducer_personalized import PersonalizedSwipeFeaturizer, resample_points, clamp

# Get hello data
import linecache
data_path = '../data/train_final_train.jsonl'
line = linecache.getline(data_path, 431621)
data = json.loads(line)
points, word = data['points'], data['word']

# Prepare features
prepared = []
for idx, pt in enumerate(points):
    raw_x = float(pt.get('x', 0.0))
    raw_y = float(pt.get('y', 0.0))
    centered_x = raw_x * 2.0 - 1.0
    centered_y = raw_y * 2.0 - 1.0
    centered_x = clamp(centered_x, -1.5, 1.5)
    centered_y = clamp(centered_y, -1.5, 1.5)
    raw_t = float(pt.get('t', idx * 10.0))
    prepared.append({'x': centered_x, 'y': centered_y, 't': raw_t})

resampled = resample_points(prepared, 82)
featurizer = PersonalizedSwipeFeaturizer(mobile_features=False)
features = featurizer(resampled)

if features.shape[1] < 37:
    padding = np.zeros((features.shape[0], 37 - features.shape[1]), dtype=np.float32)
    features = np.concatenate([features, padding], axis=1)

# Load models and metadata
with open('models/oct3_export/runtime_meta.json', 'r') as f:
    meta = json.load(f)

vocab = meta['tokens']
blank_id = meta['blank_id']
num_layers = meta['decoder_config']['num_layers']
hidden_size = meta['decoder_config']['hidden_size']

encoder_session = ort.InferenceSession('models/oct3_export/encoder.onnx')
decoder_session = ort.InferenceSession('models/oct3_export/decoder_joint.onnx')

# Run encoder
encoder_input = features.T.reshape(1, 37, -1).astype(np.float32)
encoder_outputs = encoder_session.run(None, {
    'audio_signal': encoder_input,
    'length': np.array([features.shape[0]], dtype=np.int64)
})[0]

print(f'Testing word: {word}')
print(f'Model config: {num_layers} layers, hidden_size {hidden_size}')
print(f'Encoder output shape: {encoder_outputs.shape}')

# Initialize states with correct dimensions
h_state = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
c_state = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
hypothesis = []

# Decode
for t in range(encoder_outputs.shape[2]):
    enc_frame = encoder_outputs[:, :, t:t+1]
    prev_token = np.array([[blank_id]], dtype=np.int32)
    
    for step in range(8):
        decoder_outputs = decoder_session.run(None, {
            'encoder_outputs': enc_frame,
            'targets': prev_token,
            'target_length': np.array([1], dtype=np.int32),
            'input_states_1': h_state,
            'input_states_2': c_state
        })
        
        logits = decoder_outputs[0][0, 0, 0, :]
        h_state = decoder_outputs[2]
        c_state = decoder_outputs[3]
        
        pred_id = np.argmax(logits)
        
        if pred_id == blank_id:
            break
        else:
            hypothesis.append(int(pred_id))
            prev_token = np.array([[pred_id]], dtype=np.int32)
            
            if len(hypothesis) >= 20:  # Stop if too long
                break

predicted = ''.join([vocab[i] for i in hypothesis if i < len(vocab) and vocab[i] not in ['<blank>', '<unk>']])

print(f'\nResults:')
print(f'Expected:  "{word}"')
print(f'Predicted: "{predicted}"')
print(f'Match: {predicted == word}')

# Show token IDs for debugging
print(f'\nToken IDs:')
print(f'Expected:  {[vocab.index(c) if c in vocab else -1 for c in word]}')
print(f'Predicted: {hypothesis[:len(word)+2]}')

  ONNX Export Implementation

  Based on the training script, here's what the ONNX export should look like:

  1. Model Components to Export

  The RNN-T model has three components that need exporting:
  - Encoder: Processes input features → encoded representations
  - Decoder (Prediction Network): Processes previous tokens → predictions
  - Joint Network: Combines encoder + decoder outputs → logits

  2. Expected Metadata and Vocabulary

  Based on the training script's configuration:

  {
    "vocab_size": 30,
    "blank_id": 29,
    "tokens": [
      "<blank>",  // Index 0 (placeholder, not functional blank)
      "'",        // Index 1
      "a",        // Index 2
      "b",        // Index 3
      ...
      "z",        // Index 27
      "<unk>",    // Index 28
      ""          // Index 29 (functional blank for RNN-T)
    ],
    "feature_dim": 37,
    "encoder_subsampling": 2,
    "model_type": "conformer_rnnt",
    "coordinate_system": "normalized_centered",  // [-1,1] range
    "preprocessing": {
      "resample_short_target": 56,
      "resample_long_target": 96,
      "resample_short_threshold": 48,
      "resample_long_threshold": 112
    }
  }

  3. ONNX Export Code Structure

  # Export encoder separately for streaming
  class EncoderONNX(torch.nn.Module):
      def forward(self, features: torch.Tensor, lengths: torch.Tensor):
          # features: [batch, features=37, time]
          # lengths: [batch] - actual lengths before padding
          encoded, encoded_lens = self.encoder(features, lengths)
          return encoded, encoded_lens

  # Export decoder separately
  class DecoderONNX(torch.nn.Module):
      def forward(self, targets: torch.Tensor, target_lengths: torch.Tensor, states=None):
          # targets: [batch, U] - previous predictions
          # Returns: [batch, U, hidden] predictions + updated states
          return self.decoder(targets, target_lengths, states)

  # Export joint network
  class JointONNX(torch.nn.Module):
      def forward(self, encoder_outputs: torch.Tensor, decoder_outputs: torch.Tensor):
          # Returns: [batch, T, U, vocab_size=30] logits
          return self.joint(encoder_outputs, decoder_outputs)

  4. Complete Inference Pipeline

  Here's how a swipe (x,y,t coordinates) becomes predicted words:

  def swipe_to_words(points: List[Dict], onnx_models: Dict) -> List[str]:
      """
      Complete pipeline from swipe coordinates to word predictions.

      Args:
          points: List of {"x": 0.5, "y": 0.3, "t": 100} in [0,1] range
          onnx_models: Dict with 'encoder', 'decoder', 'joint' ONNX sessions
      """

      # Step 1: Coordinate transformation [0,1] → [-1,1]
      transformed_points = []
      for pt in points:
          x_centered = pt["x"] * 2.0 - 1.0
          y_centered = pt["y"] * 2.0 - 1.0
          x_clamped = max(-1.5, min(1.5, x_centered))
          y_clamped = max(-1.5, min(1.5, y_centered))
          transformed_points.append({
              "x": x_clamped,
              "y": y_clamped,
              "t": pt["t"] - points[0]["t"]  # Relative time
          })

      # Step 2: Adaptive resampling (match training)
      num_points = len(transformed_points)
      if num_points < 48:
          target_frames = 56  # Short swipes
      elif num_points > 112:
          target_frames = 96  # Long swipes
      else:
          target_frames = num_points  # Medium swipes

      resampled = resample_to_fixed_length(transformed_points, target_frames)

      # Step 3: Feature extraction (37 dimensions)
      features = extract_features(resampled)  # Returns [37, T] array
      # Features include:
      # - Position (x, y)
      # - Velocity (dx/dt, dy/dt)
      # - Acceleration (d²x/dt², d²y/dt²)
      # - Angle, angular velocity
      # - Distance to nearest keys
      # - Spatial encoding features

      # Step 4: Run encoder
      features_tensor = np.expand_dims(features, 0)  # [1, 37, T]
      lengths = np.array([target_frames])

      encoder_out, encoded_lens = onnx_models['encoder'].run(
          None,
          {"features_bft": features_tensor, "lengths": lengths}
      )

      # Step 5: Beam search decoding with RNN-T
      beam_size = 10
      hypotheses = [([], 0.0, [29])]  # (tokens, score, last_token)

      for t in range(encoded_lens[0]):
          encoder_frame = encoder_out[0:1, t:t+1, :]  # [1, 1, hidden]
          new_hypotheses = []

          for tokens, score, last_token in hypotheses:
              # Get decoder prediction
              decoder_input = np.array([last_token])
              decoder_out = onnx_models['decoder'].run(
                  None,
                  {"targets": decoder_input}
              )[0]

              # Combine with encoder via joint network
              logits = onnx_models['joint'].run(
                  None,
                  {"encoder": encoder_frame, "decoder": decoder_out}
              )[0]  # [1, 1, 1, 30]

              probs = softmax(logits[0, 0, 0, :])

              # Extend hypotheses
              for token_id in range(30):
                  if token_id == 29:  # Blank token - don't emit
                      new_hypotheses.append((tokens, score + np.log(probs[token_id]), last_token))
                  else:
                      new_tokens = tokens + [token_id]
                      new_hypotheses.append((new_tokens, score + np.log(probs[token_id]), [token_id]))

          # Keep top-k hypotheses
          hypotheses = sorted(new_hypotheses, key=lambda x: x[1], reverse=True)[:beam_size]

      # Step 6: Convert token IDs to words
      vocab = ["<blank>", "'", "a", "b", ..., "z", "<unk>", ""]

      predicted_words = []
      for tokens, score, _ in hypotheses[:5]:  # Top 5 predictions
          chars = []
          for tok_id in tokens:
              if 1 <= tok_id <= 27:  # Valid character
                  chars.append(vocab[tok_id])
          word = "".join(chars)
          predicted_words.append((word, score))

      return predicted_words

  5. Key Implementation Details

  Critical for ONNX export:
  - Separate models: Export encoder, decoder, and joint as separate ONNX files for flexibility
  - Dynamic axes: Support variable sequence lengths for both time and batch dimensions
  - State handling: Decoder might need state for streaming (LSTM hidden states)
  - Blank token: Index 29 is the functional blank, not index 0
  - Output shape: Joint network outputs [batch, T, U, vocab_size=30]

  Runtime requirements:
  - Feature extraction must match training (37-dim features)
  - Coordinate transformation [0,1] → [-1,1] is mandatory
  - Adaptive resampling logic must match training
  - Beam search with RNN-T-specific blank handling

  This gives you a complete, working inference pipeline from raw swipe coordinates to predicted words using the exported ONNX models!

  do the current files @trained_models/nema1/export_stateful_pair.py and @web-demo/js/feature-extractor-corrected.js and @web-demo/js/onnx-rnnt-decoder-fixed.js align with your
  envisioned implementation?
  ⎿  Read trained_models/nema1/export_stateful_pair.py (123 lines)
  ⎿  Read web-demo/js/feature-extractor-corrected.js (215 lines)
  ⎿  Read web-demo/js/onnx-rnnt-decoder-fixed.js (371 lines)

● Looking at the actual implementation files, they align quite well with what I envisioned, with a few important observations:

  ✅ export_stateful_pair.py - Mostly Correct

  This export script correctly:
  - Uses NeMo's built-in model.export() which produces encoder.onnx and decoder_joint.onnx
  - Extracts vocabulary and blank_id from the model
  - Creates runtime_meta.json with the correct structure

  Key insight: NeMo handles the RNN-T export by combining decoder + joint network into a single decoder_joint.onnx file, which is actually more efficient than separate models.

  ⚠️ feature-extractor-corrected.js - Has Issues

  Problems:
  1. Coordinate confusion: Lines 42-64 try to auto-detect if coordinates are in [-1,1] or [0,1], but then stores them in [0,1] range (lines 63-64). This is backwards from
  training!
  2. No resampling applied: Line 191 comments say "no resampling" but training uses adaptive resampling (56-96 frames)
  3. Missing transformation: Should transform [0,1] → [-1,1] for model input, not keep in [0,1]

  Should be:
  // Transform from [0,1] input to [-1,1] for model
  const centeredX = x * 2.0 - 1.0;
  const centeredY = y * 2.0 - 1.0;

  ✅ onnx-rnnt-decoder-fixed.js - Excellent Implementation

  This decoder is well-implemented:
  - Correctly handles the combined decoder_joint.onnx model
  - Proper RNN-T decoding with blank token at index 29
  - Stateful LSTM handling (lines 145-146, 189-190)
  - Both greedy and beam search with lexicon constraints
  - Correct feature transposition for encoder input (lines 122-127)

  Particularly good:
  - Lines 173-204: Proper RNN-T greedy decoding that emits multiple symbols per frame
  - Lines 258-368: Sophisticated beam search with trie-based lexicon constraints

  Summary of Alignment

  | Component    | Expected                       | Actual                 | Status   |
  |--------------|--------------------------------|------------------------|----------|
  | Model Export | Separate encoder/decoder/joint | Combined decoder_joint | ✅ Better |
  | Metadata     | vocab_size=30, blank_id=29     | Correct                | ✅        |
  | Coordinates  | [0,1] → [-1,1] transform       | Confused/backwards     | ❌        |
  | Resampling   | Adaptive 56-96 frames          | Missing                | ❌        |
  | Features     | 37 dimensions                  | Correct                | ✅        |
  | Decoding     | RNN-T with blank=29            | Correct                | ✅        |
  | Beam Search  | Lexicon-constrained            | Well implemented       | ✅        |

  Critical Fix Needed

  The feature-extractor-corrected.js needs to:
  1. Accept [0,1] coordinates (keyboard space)
  2. Transform to [-1,1] (model space)
  3. Apply adaptive resampling to match training

  // Fixed coordinate transformation
  normalizePoints(points) {
      // Input: [0,1] coordinates from keyboard
      // Output: [-1,1] coordinates for model
      return points.map(pt => ({
          x: pt.x * 2.0 - 1.0,  // Transform [0,1] → [-1,1]
          y: pt.y * 2.0 - 1.0,
          t: pt.t - points[0].t
      }));
  }

  The export and decoder are solid, but the feature extractor's coordinate handling will cause inference failures if not fixed!
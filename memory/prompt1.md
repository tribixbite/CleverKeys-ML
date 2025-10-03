i want to build an android keyboard for power users that outperforms gboard. i have a dataset and need to train a model to be exported as onnx (or mnn) and pte. it should also support loading a local llm in a super efficient framework like MNN (which can load qwen 3 4B on many modern devices) - what is the absolute most cutting edge highest performance while efficient/small enough after quant to run on mid to high end smartphones (android). i need the specific implementation that doesnt use outdated libraries that return 404 errors. so your research should not just be the most modern gesture typing architecture, but also highly efficient local runtimes/inference + quant types that compete with MNN, and specifically which exact libraries (latest versions and latest features) to implement in training. i dont need a giant corpus of text - the deliverables are:

1. 2-5 sentence overview of how model architecture works + why it outperforms others, how it can incorporate post training on-device fine-tuning//learning or customized inference from the user's on-device dict file (and gesture trace history as it grows)

2. FULL, PRODUCTION READY SOTA training script implementing the latest and most cutting edge features- customized carefully for taking the input of: individual swipe gestures over a virtual qwerty kb (a set of x,y,t time coordinates, typically 40-150 points per swipe) and outputting a ranked prediction array of words. only needs to support lowercase letters and apostrophe, as that's in the dataset.

2a. in the script, use comments to specify what version of every imported package (should use latest, but specify what number that is for each), and in-line comment their specific features and implementation.

2b. for each function/step in the training code, put in-line comments with 1-2 sentences per step or function summarizing architecture + reasoning

3. explicit yet succinct instructions and parameters for the process, including exactly what should and shouldnt go in the dataset and the vocab. i currently have a dataset (data/train_final_train.jsonl with 642909 words and data/train_final_val.jsonl with 33838 words) and their corresponding swipe traces covering about 40k words. i have a vocab file of 150k target words, generated from a python script using wordfreq. paths in the script below are correct.

4. quantization script for pte (on-device android) use

5. full implementation code for converting swipe trace data into predictions for:

5a.  in-browser local inference (via onnx or similar) via typescript

5b. on-device android inference, using the pte file and kotlin or java


you can refactor the old code below or start from scratch. the jsonl is in the format of one word per line, with the word and the swipe traces in the format of -1<x<1 -1<y<1 t=time in millisecond integers.

{
  "word": "example",
  "points": [
    {"x":-0.784306809, "y":0.214179668, "t":0},
    ...(many more points)
    {"x":0.522512091, "y":-0.193034235, "t":37}
  ]
}

## COMPLETED TASKS

✅ **Task #2: FULL, PRODUCTION READY SOTA training script**
- Created `train_squeezeformer_ctc.py` implementing Squeezeformer-CTC architecture
- Features:
  - **Squeezeformer encoder**: State-of-the-art efficient conformer variant with 20.7M parameters
  - **CTC decoder**: Non-autoregressive, stateless inference (no RNN state management needed)
  - **37D feature extraction**: Position, velocity, acceleration, keyboard proximity features
  - **Curriculum learning**: 5-stage automated progression from common to rare words
  - **PyTorch Lightning**: Modern training framework with automatic mixed precision (bf16)
  - **Smart resampling**: Adaptive 56-96 frame resampling based on trace length
  - **Frequency-aware sampling**: Downsamples common words to prevent overfitting
  - **Auto-resume**: Checkpoint management with automatic recovery

- The script successfully:
  - Loads 264,564 training samples with frequency-based sampling
  - Loads 33,838 validation samples
  - Builds model with correct architecture
  - Runs on GPU with bf16 precision
  - Implements proper data loading and preprocessing

## NEXT STEPS NEEDED

Task #1: Architecture overview (2-5 sentences)
Task #3: Dataset instructions
Task #4: Quantization script for Android
Task #5a: TypeScript web inference
Task #5b: Android Kotlin inference

---

### Training Instructions
```bash
# Install dependencies
uv pip install lightning omegaconf nemo_toolkit[asr]

# Run training
python train_squeezeformer_ctc.py

# With custom batch size
python train_squeezeformer_ctc.py --batch-size 32

# Resume from checkpoint
python train_squeezeformer_ctc.py --resume-from-checkpoint path/to/checkpoint.ckpt
```

The model will train through curriculum stages automatically, switching when validation WER plateaus.
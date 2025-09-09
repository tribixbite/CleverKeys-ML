This document outlines the architecture and usage of the provided
`train_production.py` script. This pipeline is designed from the ground
up for performance, accuracy, and deployability on mobile devices.

1. Architecture
-------------------------------------------

**Transformer Encoder with a Connectionist Temporal Classification (CTC)
loss function.**

This is a critical, deliberate choice for a production system for
several reasons:

-   **Inference Speed:** CTC models are non-autoregressive. The entire
    output sequence is predicted in a single forward pass, making
    inference significantly faster than decoder-based models which must
    generate one token at a time. Speed is paramount for a real-time
    keyboard.

-   **Simplicity:** The model architecture is simpler, containing only
    an encoder. This results in a smaller model size and fewer points of
    failure.

-   **Alignment-Free:** CTC loss cleverly handles the alignment between
    the long gesture sequence (e.g., 100 points) and the short character
    sequence (e.g., 4 characters for \"what\"). We don\'t need to
    pre-process the data to create explicit alignments.

2. The Three Core Components
----------------------------

Our pipeline consists of three modular stages:

### A. Feature Engineering (`SwipeFeaturizer`)

The raw list of `(x, y, t)` points is transformed into a rich,
high-dimensional feature vector for every point in the trace. This
includes:

-   **Kinematics:** Velocity and acceleration, which capture the
    *dynamics* of the user\'s motion.

-   **Spatial Context:** A one-hot encoded vector of the nearest
    physical key on the keyboard, giving the model crucial information
    about where the user\'s finger is.

### B. The Neural Model (`GestureCTCModel`)

This is the brain of the operation. It\'s a standard Transformer Encoder
that reads the sequence of feature vectors and outputs a probability
distribution over all possible characters (a-z, \') for each time step.
Its sole job is to learn the complex relationship between gesture shapes
and character likelihoods.

### C. The Decoder (`pyctcdecode`)

This is the secret weapon for achieving high accuracy. The raw output
from the neural model is a \"noisy\" sequence of probabilities. A simple
greedy decoding would produce many non-words (e.g., \"whst\").

Instead, we use `pyctcdecode`, a highly optimized beam search decoder
that integrates two powerful constraints:

1.  **Vocabulary Trie:** It uses your 153k-word vocabulary list to build
    a prefix tree (trie). This instantly prunes any search path that
    does not lead to a valid word in your vocabulary.

2.  **Language Model (KenLM):** We load a pre-trained N-gram language
    model (downloaded automatically from Hugging Face). This guides the
    search towards more common and plausible words. For example, if the
    gesture is ambiguous between \"hat\" and \"what\", the LM score for
    \"what\" will be much higher, leading to the correct prediction.

**This \"strong decoding\" process is how you achieve state-of-the-art
results without needing an infinitely large and complex neural model.**

3. Supporting User-Specific Vocabulary
--------------------------------------

You wanted to support learning a user\'s custom words without retraining
the core model. This architecture is **perfect** for that.

The neural model and the decoder are **decoupled**. The neural model
(`.pte` file) is a static asset in your app. The decoder\'s vocabulary,
however, can be dynamic.

**At inference time in your Android app, you can:**

1.  Maintain a list of words the user has typed that are not in the main
    dictionary.

2.  Dynamically add these new words to the `pyctcdecode` vocabulary trie
    **on the fly**.

3.  The next time the user swipes a similar gesture, the decoder will
    now consider their custom word as a valid candidate, allowing the
    keyboard to \"learn\" their vocabulary over time.

4. How to Use This Pipeline
---------------------------

### Step 1: Setup

Install the necessary libraries:

    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    pip install tqdm numpy huggingface_hub
    pip install pyctcdecode==0.4.0 # Use a specific version for stability
    pip install [https://github.com/kpu/kenlm/archive/master.zip](https://github.com/kpu/kenlm/archive/master.zip)
    pip install tensorboard

### Step 2: Configure

Open `train_production.py` and modify the `CONFIG` dictionary at the
top. **You must update the paths** to your training data, validation
data, and vocabulary file. You can also tune hyperparameters here.

### Step 3: Train

Run the training script from your terminal:

    python train_production.py

Training progress will be logged to the console, and detailed metrics
will be saved in the `logs/` directory. You can view them by running:

    tensorboard --logdir logs

### Step 4: Export & Deploy

After training, the best model checkpoint (`best_model.pth`) and
exported versions (`model.pt`, `model.onnx`) will be saved in the
`checkpoints/` and `exports/` directories respectively.

-   **For Android:** Take the `exports/model.pt` file and use the
    [ExecuTorch
    toolchain](https://pytorch.org/executorch/stable/getting-started-setup.html "null")
    to convert it to the final `.pte` format for your app.

-   **For Browser:** Use the `exports/model.onnx` file with a runtime
    like ONNX.js.


### Step 5: Deployment & Export Guide

The training script saves two key artifacts in the `exports/` directory:
`model.pt` (TorchScript) and `model.onnx`. This guide explains how to
convert these into deployable assets for your target platforms.

#### A. For Android (via ExecuTorch `.pte`)

The goal is to convert the `model.pt` file into a highly-optimized
`.pte` file. This is a multi-step process using the ExecuTorch
toolchain.

**Prerequisites:** You must have the ExecuTorch environment set up.
Follow the official [ExecuTorch setup
guide](https://pytorch.org/executorch/stable/getting-started-setup.html "null")
to build the necessary compiler tools.

**Conversion Steps:**

1.  **Lower to EDGE Dialect:** The first step converts the TorchScript
    model into an intermediate representation (IR) that ExecuTorch
    understands.

        # Run this from your terminal after setting up the ExecuTorch SDK
        python -m executorch.exir.capture --model_path exports/model.pt --output_path exports/model.edge.pt

2.  **Convert to `.pte` format:** The final step takes the EDGE model
    and compiles it into the lean `.pte` flatbuffer format.

        # This command generates the final deployable file
        executorch-compiler -m exports/model.edge.pt -o exports/model.pte

**Using it in your Android App:**

1.  **Add the `.pte` file:** Copy the generated `exports/model.pte` into
    your Android app\'s `assets` directory.

2.  **Add ExecuTorch Runtime Dependency:** In your app\'s `build.gradle`
    file, add the dependency for the ExecuTorch runtime.

        // In your app/build.gradle dependencies block
        implementation 'org.pytorch:executorch:0.1.0' 

3.  **Load and run the model:** In your Kotlin/Java code, you can now
    load the model from assets and run inference.

        // Example Kotlin code snippet
        val module = Module.load(assetFilePath("model.pte"))
        val inputTensor = Tensor.fromBlob(yourFeatureData, longArrayOf(1, seqLen, featureDim))
        val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()
        val logProbs = outputTensor.dataAsFloatArray

#### B. For Web Browsers (via ONNX)

The `model.onnx` file is ready to be used directly by a JavaScript
runtime. The most common one is `ONNX.js`.

**Using it in your Web App:**

1.  **Include the runtime:** Add the ONNX.js library to your HTML file
    via a script tag.

        <script src="[https://cdn.jsdelivr.net/npm/onnxjs/dist/onnx.min.js](https://cdn.jsdelivr.net/npm/onnxjs/dist/onnx.min.js)"></script>

2.  **Load and run the model:** In your JavaScript code, create an
    inference session and run the model.

        // Example JavaScript code snippet
        async function runInference(featureData) {
            // create a session
            const session = new onnx.InferenceSession();
            // load the ONNX model.
            await session.loadModel("./model.onnx");

            // create a tensor for the input
            const inputTensor = new onnx.Tensor(featureData, 'float32', [1, featureData.length, featureData[0].length]);

            // create a tensor for the padding mask (all false/0 for a single trace)
            const maskData = new Float32Array(featureData.length).fill(0);
            const maskTensor = new onnx.Tensor(maskData, 'float32', [1, featureData.length]);

            const outputMap = await session.run([inputTensor, maskTensor]);
            const outputTensor = outputMap.get('log_probs'); // Name must match export

            console.log('Model output:', outputTensor.data);
            // This output would then be fed to a JavaScript CTC decoder
        }



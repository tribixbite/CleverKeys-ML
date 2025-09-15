from huggingface_hub import list_models, list_repo_files

# Search for KenLM models
print("Searching for KenLM models...")
models = list(list_models(search='kenlm', limit=30))
for m in models:
    print(f"\n{m.modelId}")
    try:
        files = list_repo_files(m.modelId)
        arpa_files = [f for f in files if '.arpa' in f and '.bin' not in f and '.gz' not in f]
        if arpa_files:
            print(f"  ARPA files: {arpa_files[:5]}")
    except:
        pass

# Also check some known speech recognition models
print("\n\nChecking speech recognition models with LM...")
speech_models = [
    "patrickvonplaten/wav2vec2-base-100h-with-lm",
    "jonatasgrosman/wav2vec2-large-xlsr-53-english",
    "facebook/wav2vec2-base-960h"
]

for model_id in speech_models:
    try:
        print(f"\n{model_id}:")
        files = list_repo_files(model_id)
        lm_files = [f for f in files if 'lm' in f.lower() or 'arpa' in f.lower() or 'kenlm' in f.lower()]
        if lm_files:
            print(f"  LM files: {lm_files[:5]}")
    except:
        print(f"  Could not access")
#!/usr/bin/env python3
import requests
import os

print("Downloading KenLM model from edugp/kenlm...")
url = "https://huggingface.co/edugp/kenlm/resolve/main/wikipedia/en.arpa.bin"
output_path = "/tmp/en_wikipedia.arpa.bin"

response = requests.get(url, stream=True)
total_size = int(response.headers.get('content-length', 0))

with open(output_path, 'wb') as f:
    downloaded = 0
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"\rProgress: {percent:.1f}%", end="")

print(f"\nModel saved to {output_path}")
print(f"File size: {os.path.getsize(output_path) / (1024**3):.2f} GB")
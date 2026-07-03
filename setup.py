import os
from datasets import load_dataset
from huggingface_hub import snapshot_download

cwd = os.getcwd()
print("starting download of the gsm8k dataset...")
data = load_dataset("openai/gsm8k", "main")
data.save_to_disk(os.path.join(cwd, "gsm8k_dataset"))
print("finished download :D")

print(
    "starting download of the base model which we are gonna be post-training from scratch!"
)
snapshot_download(
    repo_id="Qwen/Qwen2.5-1.5B", local_dir=os.path.join(cwd, "Qwen2.5-1.5B base model")
)
print("finished download :D")

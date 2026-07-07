import os
from datasets import load_dataset
from huggingface_hub import snapshot_download

cwd = os.getcwd()
print("starting download of the MetaMathQA dataset for the initial sft phase...")
data = load_dataset("meta-math/MetaMathQA", split="train")
data.to_json(os.path.join(cwd, "metamathqa.json"))
print("finished download :D")

print(
    "starting download of the base model which we are gonna be post-training from scratch!"
)
snapshot_download(
    repo_id="Qwen/Qwen2.5-1.5B", local_dir=os.path.join(cwd, "Qwen2.5-1.5B base model")
)
print("finished download :D")

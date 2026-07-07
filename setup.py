# %%
# downloads
import os
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

cwd = os.getcwd()
print("starting download of the MetaMathQA dataset for the initial sft phase...")
data = load_dataset("meta-math/MetaMathQA", split="train")
data.save_to_disk(os.path.join(cwd, "metamathqa"))
print("finished download :D")

print(
    "starting download of the base model which we are gonna be post-training from scratch!"
)
snapshot_download(
    repo_id="Qwen/Qwen2.5-1.5B", local_dir=os.path.join(cwd, "Qwen2.5-1.5B base model")
)
print("finished download :D")

# %%
# processing
tokeniser = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
data[0]
data[1]
data[2]


def process(entry):
    parts = entry["response"].rsplit("The answer is:", 1)
    thinking = parts[0].strip()
    answer = parts[1].strip()
    return f""""<|im_start|>system
    You are a helpful assistant. You must think step-by-step inside <think> tags before providing the final answer after ####.<|im_end|>
    <|im_start|>user
    {entry["query"]}<|im_end|>
    <|im_start|>assistant
    <think>
    {thinking}
    </think>
    #### {answer}<|im_end|>"""


processed_data = data.map(process, batch_size=True, remove_columns=data.column_names)

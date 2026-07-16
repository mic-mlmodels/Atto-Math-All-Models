# %%
# imports
from transformers import AutoTokenizer
from utils import extract_answer, load_cooked_model
import os
import torch
from dataloader import Dataloader
from datasets import load_dataset

# %%
# setup
MAX_TOKENS = 768
BATCH_SIZE = 1
BOTTNECK_RANK = 16
LORA_ALPHA = BOTTNECK_RANK * 2
NUM_STEPS = 15000
MAX_LR = 1e-4
MIN_LR = 1e-5
device = "cuda" if torch.cuda.is_available() else "cpu"
cwd = os.getcwd()
gsm8k_test = load_dataset("openai/gsm8k", "main", split="test")
gsm8k_test.save_to_disk(os.path.join(cwd, "gsm8k_test"))


model = load_cooked_model(
    BOTTNECK_RANK,
    LORA_ALPHA,
    device,
    params_path=cwd + "/Atto-Math-SFT-V0-checkpoint1.pt",
)
tokeniser = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
dataloader = Dataloader(gsm8k_test, True, tokeniser, BATCH_SIZE)
data_iter = iter(dataloader)
correct = 0
total = 0
# %%
# actual eval section

model.eval()
model.to(device)  # type: ignore
with torch.inference_mode():
    for i in range(len(dataloader)):
        if i % 8 == 0:
            print("i")
        param_dict = next(data_iter)
        param_dict = {k: v.to(device) for k, v in param_dict.items() if k != "labels"}
        out = extract_answer(model(**param_dict))
        if out == extract_answer(param_dict["labels"]):
            correct += 1
        total += 1
        del out, param_dict

# %%
# results
print(correct, total)

# %%
# data testing grounds
gsm8k_test[0]

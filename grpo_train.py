# %%
# imports

import torch.nn.functional as F
from transformers import AutoTokenizer
from utils import extract_answer, load_cooked_model, eval_process
import os
import torch
from dataloader import Dataloader
from datasets import load_dataset, load_from_disk
import bitsandbytes as bnb

# %%
# setup

MAX_TOKENS = 768
EVAL_MAJ_BATCH_SIZE = 8
BATCH_SIZE = 16
BOTTNECK_RANK = 16
LORA_ALPHA = BOTTNECK_RANK * 2
NUM_STEPS = 15000
MAX_LR = 1e-4
MIN_LR = 1e-5
device = "cuda" if torch.cuda.is_available() else "cpu"
cwd = os.getcwd()
data = load_from_disk("processed-metamathqa")
data = data.filter(
    lambda x: len(x["input_ids"]) <= MAX_TOKENS
)  # maybe should have put this in the initial data processing setup but this keeps it more flexible if i need to change filtering for grpo for example
data = data.train_test_split(test_size=0.1, train_size=0.9)  # type: ignore
train_data = data["train"]
val_data = data["test"]
tokeniser = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
model = load_cooked_model(
    BOTTNECK_RANK,
    LORA_ALPHA,
    params_path=cwd + "/Atto-Math-SFT-V0-checkpoint1.pt",
)
model.config.use_cache = False
model.enable_input_require_grads()
model.gradient_checkpointing_enable()
train_dataloader = Dataloader(train_data, True, tokeniser, BATCH_SIZE)
val_dataloader = Dataloader(val_data, False, tokeniser, BATCH_SIZE)
train_iter = iter(train_dataloader)
val_iter = iter(val_dataloader)
model.to(device)  # type: ignore
optimiser = bnb.optim.PagedAdamW8bit(  # type: ignore
    params=[param for param in model.parameters() if param.requires_grad],
    lr=MAX_LR,
)

# %%
# grpo time fellas ;D

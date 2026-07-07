# %%
# imports
import torch
from datasets import load_from_disk, Dataset
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from dataloader import Dataloader

data = load_from_disk("processed-metamathqa")

data = data.train_test_split(test_size=0.1, train_size=0.9)  # type: ignore
train_data = data["train"]
val_data = data["test"]

# %%
# base model

tokeniser = AutoTokenizer.from_pretrained("Qwen2.5-1.5B base model")
model = AutoModelForCausalLM.from_pretrained("Qwen2.5-1.5B base model")

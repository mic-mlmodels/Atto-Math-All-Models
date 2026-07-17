# %%
# imports
import os
import torch
from transformers import AutoModelForCausalLM
from qlora import adapt_model


# %%
# func to return the model with my glorious params
def load_cooked_model(BOTTNECK_RANK, LORA_ALPHA, device, params_path):
    model = AutoModelForCausalLM.from_pretrained("Qwen2.5-1.5B base model")
    for param in model.parameters():
        param.requires_grad = False
    adapt_model(model, BOTTNECK_RANK, LORA_ALPHA)
    model.load_state_dict(
        torch.load(params_path, weights_only=False)["model_state_dict"], strict=False
    )  # for anyone running this code (thx for checking it out btw), i set the weights_only param as False but this is safe as its just primarily to allow some numpy ops in the pickle stream and nothing malicious
    return model


# %%
# func to extract out the actual answer
def extract_answer(out):
    return out.rsplit("####", 1)[-1].split("<|im_end|>")[0].split("<|endoftext|>")[0].strip()


# %%
# top secret testing ground
MAX_TOKENS = 768
BATCH_SIZE = 2
BOTTNECK_RANK = 16
LORA_ALPHA = BOTTNECK_RANK * 2
NUM_STEPS = 15000
MAX_LR = 1e-4
MIN_LR = 1e-5
device = "cuda" if torch.cuda.is_available() else "cpu"
cwd = os.getcwd()


model = load_cooked_model(
    BOTTNECK_RANK,
    LORA_ALPHA,
    device,
    params_path=cwd + "/Atto-Math-SFT-V0-checkpoint1.pt",
)

# %%
# top secret testing ground

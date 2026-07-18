# %%
# imports
import os
import torch
from transformers import AutoModelForCausalLM
from qlora import adapt_model


# %%
# func to return the model with my glorious params
def load_cooked_model(BOTTNECK_RANK, LORA_ALPHA, params_path):
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
    return (
        out.rsplit("####", 1)[-1]
        .split("<|im_end|>")[0]
        .split("<|endoftext|>")[0]
        .strip()
    )


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

# %%
# eval process


def eval_process(entries, tokeniser):
    input_ids = []
    attention_mask = []
    labels = []
    for response, query in zip(entries["answer"], entries["question"]):
        if "####" in response:
            parts = response.rsplit("####", 1)
            labels.append(parts[1].strip())
            tokens = tokeniser(
                tokeniser.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant. You must think step-by-step inside <think> tags before providing the final answer after ####.",
                        },
                        {"role": "user", "content": query},
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            )["input_ids"]
            input_ids.append(tokens)
            attention_mask.append([1] * len(tokens))

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}  # type: ignore

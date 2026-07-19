# %%
# imports
import torch.nn.functional as F
from transformers import AutoTokenizer
from utils import extract_answer, load_cooked_model, eval_process
import os
import torch
from dataloader import Dataloader
from datasets import load_dataset

# %%
# setup
MAX_TOKENS = 768
EVAL_MAJ_BATCH_SIZE = 8
BATCH_SIZE = 32
BOTTNECK_RANK = 16
LORA_ALPHA = BOTTNECK_RANK * 2
NUM_STEPS = 15000
MAX_LR = 1e-4
MIN_LR = 1e-5
device = "cuda" if torch.cuda.is_available() else "cpu"
cwd = os.getcwd()
tokeniser = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
gsm8k_test = load_dataset("openai/gsm8k", "main", split="test")
gsm8k_test.save_to_disk(os.path.join(cwd, "gsm8k_test"))
processed_data = gsm8k_test.map(
    eval_process,
    batched=True,
    remove_columns=gsm8k_test.column_names,
    fn_kwargs={"tokeniser": tokeniser},
)
processed_data.save_to_disk(os.path.join(cwd, "processed-gsm8k-test"))

len(processed_data)

model = load_cooked_model(
    BOTTNECK_RANK,
    LORA_ALPHA,
    params_path=cwd + "/Atto-Math-SFT-V0-checkpoint1.pt",
)
dataloader = Dataloader(processed_data, True, tokeniser, BATCH_SIZE, eval=True)
data_iter = iter(dataloader)
correct = 0
total = 0
# %%
# actual eval section

model.eval()
model.to(device)  # type: ignore
with torch.inference_mode():
    for i in range(len(dataloader)):
        print(i)
        original_param_dict = next(data_iter)
        current_batch = original_param_dict["input_ids"].shape[0]
        mask = (
            original_param_dict["attention_mask"]
            .repeat_interleave(EVAL_MAJ_BATCH_SIZE, dim=0)
            .to(device)
        )
        input = (
            original_param_dict["input_ids"]
            .repeat_interleave(EVAL_MAJ_BATCH_SIZE, dim=0)
            .to(device),
            mask,
        )
        tokenised_prompt = (
            original_param_dict["input_ids"]
            .repeat_interleave(EVAL_MAJ_BATCH_SIZE, dim=0)
            .to(device)
        )
        finished = torch.zeros(
            current_batch * EVAL_MAJ_BATCH_SIZE,
            dtype=torch.bool,
        ).to(device)
        imend_token = tokeniser.convert_tokens_to_ids("<|im_end|>")
        kv_cache = None
        while not finished.all() and tokenised_prompt.shape[-1] < 1024:
            out = model(
                *input, past_key_values=kv_cache, use_cache=True, logits_to_keep=1
            )
            kv_cache = out.past_key_values
            logits = out.logits
            probs = F.softmax(logits[:, -1, :], dim=-1)
            dist_obj = torch.distributions.Categorical(probs)
            next_word = dist_obj.sample()
            mask = torch.cat(
                (
                    mask,
                    torch.ones(EVAL_MAJ_BATCH_SIZE * current_batch, 1, device=device),
                ),
                dim=-1,
            )
            input = (next_word.unsqueeze(1), mask)
            finished = (
                finished
                | (next_word == tokeniser.eos_token_id)
                | (next_word == imend_token)
            )
            tokenised_prompt = torch.cat(
                (tokenised_prompt, torch.unsqueeze(next_word, dim=1)),
                dim=-1,
            )
        decoded_out = tokeniser.batch_decode(tokenised_prompt)
        for i in range(current_batch):
            group_correct = 0
            maj_dict = {}
            for row in decoded_out[
                i * EVAL_MAJ_BATCH_SIZE : i * EVAL_MAJ_BATCH_SIZE + EVAL_MAJ_BATCH_SIZE
            ]:
                try:
                    if float(extract_answer(row)) == float(
                        original_param_dict["labels"][i]  # type: ignore
                    ):  # type: ignore
                        group_correct += 1
                    else:
                        maj_dict[extract_answer(row)] = 1 + maj_dict.get(
                            extract_answer(row), 0
                        )
                except ValueError:
                    # print(
                    #     extract_answer(row)
                    # )  # temp btw just to see what types of questions my model fail on.
                    maj_dict[extract_answer(row)] = 1 + maj_dict.get(
                        extract_answer(row), 0
                    )
            highest = 0
            for k, v in maj_dict.items():
                if v > highest:
                    highest = v
            if (
                group_correct > highest
            ):  # its kinda ambigious what the standard for maj is so i currently set it as beating the mode and not just mode itself
                correct += 1
            total += 1
        del out, original_param_dict  # type: ignore

# %%
# results
print(correct, total)

# %%
# data testing grounds
gsm8k_test[0]
processed_data[0]

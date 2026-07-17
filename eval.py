# %%
# imports
import torch.nn.functional as F
from transformers import AutoTokenizer
from utils import extract_answer, load_cooked_model
import os
import torch
from dataloader import Dataloader
from datasets import load_dataset
from setup import eval_process

# %%
# setup
MAX_TOKENS = 768
EVAL_BATCH_SIZE = 8
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
processed_data = gsm8k_test.map(
    eval_process, batched=True, remove_columns=gsm8k_test.column_names
)
processed_data.save_to_disk(os.path.join(cwd, "processed-gsm8k-test"))


model = load_cooked_model(
    BOTTNECK_RANK,
    LORA_ALPHA,
    device,
    params_path=cwd + "/Atto-Math-SFT-V0-checkpoint1.pt",
)
tokeniser = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
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
        if i % 8 == 0:
            print(i)
        original_param_dict = next(data_iter)
        tokenised_prompt = (
            original_param_dict["input_ids"].repeat(EVAL_BATCH_SIZE, 1).to(device)
        )
        finished = torch.zeros(
            EVAL_BATCH_SIZE,
            dtype=torch.bool,
        ).to(device)
        imend_token = tokeniser.convert_tokens_to_ids("<|im_end|>")
        while not finished.all() and tokenised_prompt.shape[-1] < 1024:
            out = model(tokenised_prompt)
            logits = out.logits
            probs = F.softmax(logits[:, -1, :], dim=-1)
            dist_obj = torch.distributions.Categorical(probs)
            next_word = dist_obj.sample()
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
            group_correct = 0
            # THIS IS WRONG IMPLENTATION OF MAJ BUT I GOTTA FIX UP EVERYTHING ELSE FIRST
            for row in decoded_out:
                if int(extract_answer(row)) == int(original_param_dict["labels"]):  # type: ignore
                    group_correct += 1
            if group_correct > 4:
                correct += 1
            total += 1
            del out  # type: ignore
            # THIS IS WRONG IMPLENTATION OF MAJ BUT I GOTTA FIX UP EVERYTHING ELSE FIRST
        del original_param_dict
# %%
# results
print(correct, total)

# %%
# data testing grounds
gsm8k_test[0]
processed_data[0]

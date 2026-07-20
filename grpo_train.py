# %%
# imports
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer
from utils import extract_answer, load_cooked_model
import os
import torch
from dataloader import Dataloader
from datasets import load_from_disk
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
optimiser = bnb.optim.PagedAdamW8bit(  # type: ignore
    params=[param for param in model.parameters() if param.requires_grad],
    lr=MAX_LR,
)

# %%
# grpo time fellas ;D

EPISODE_NUM = 1000
DISCOUNT = 0.99
EPSILON = 0.2
OLD_POLICY_LOOPS = 4
NEW_POLICY_LOOPS = 8
new_policy_v0 = model.to(device)  # type: ignore
old_policy_v0 = model.to(device)  # type: ignore
policy_optimiser = torch.optim.AdamW(lr=3e-4, params=new_policy_v0.parameters())
episode_rewards = []
mean_rewards = []
for episode in range(EPISODE_NUM):
    if episode % 100 == 0:
        print(episode)
    old_policy_v0.load_state_dict(new_policy_v0.state_dict())
    for param in old_policy_v0.parameters():
        param.requires_grad = False
    old_log_probs_stack = []
    old_returns_stack = []
    tokenised_prompt_stack = []
    with torch.no_grad():
        for i in range(OLD_POLICY_LOOPS):
            print(i)
            original_param_dict = next(train_iter)
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
            old_log_probs_lst = []
            while not finished.all() and tokenised_prompt.shape[-1] < 1024:
                out = old_policy_v0(
                    *input, past_key_values=kv_cache, use_cache=True, logits_to_keep=1
                )
                kv_cache = out.past_key_values
                logits = out.logits
                probs = F.softmax(logits[:, -1, :], dim=-1)
                dist_obj = torch.distributions.Categorical(probs)
                next_word = dist_obj.sample()
                old_log_probs_lst.append(dist_obj.log_prob(next_word))
                mask = torch.cat(
                    (
                        mask,
                        torch.ones(
                            EVAL_MAJ_BATCH_SIZE * current_batch, 1, device=device
                        ),
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
            old_log_probs_tensor = torch.stack(old_log_probs_lst)
            old_log_probs_stack.append(old_log_probs_tensor)
            for i in range(current_batch):
                group_correct = 0
                maj_dict = {}
                for row in decoded_out[
                    i * EVAL_MAJ_BATCH_SIZE : i * EVAL_MAJ_BATCH_SIZE
                    + EVAL_MAJ_BATCH_SIZE
                ]:
                    try:
                        if float(extract_answer(row).replace(",", "")) == float(
                            original_param_dict["labels"][i].replace(",", "")  # type: ignore
                        ):
                            old_returns_tensor = torch.ones_like(
                                old_log_probs_tensor, device=device
                            )
                        else:
                            old_returns_tensor = torch.zeros_like(
                                old_log_probs_tensor, device=device
                            )
                    except ValueError:
                        print("Value error oh no")
                        old_returns_tensor = torch.zeros_like(
                            old_log_probs_tensor, device=device
                        )
            tokenised_prompt_stack.append(tokenised_prompt)
            old_returns_stack.append(old_returns_tensor)  # type: ignore
            del out, original_param_dict, kv_cache  # type: ignore
    old_advantage_tensor = torch.stack(old_returns_stack) - torch.mean(
        torch.stack(old_returns_stack), dim=-1
    )  # type: ignore
    old_log_probs_stack = torch.stack(old_log_probs_stack)
    tokenised_prompt_stack = torch.stack(tokenised_prompt_stack)
    # UP TO HERE BTW JUST START IMPLEMENTING NEW POLICY LOOP
    for i in range(NEW_POLICY_LOOPS):
        kv_cache = None
        out = new_policy_v0(
            input_ids=tokenised_prompt_stack, past_key_values=kv_cache, use_cache=True
        )
        kv_cache = out.past_key_values
        logits = out.logits
        log_probs = F.log_softmax(logits, dim=-1)
        targets = tokenised_prompt_stack[:, :, 1:]
        new_log_probs = log_probs.gather(-1, targets)
        policy_optimiser.zero_grad()
        policy_loss = -torch.mean(
            torch.minimum(
                torch.exp(new_log_probs - old_log_probs_stack) * old_advantage_tensor,
                torch.clip(
                    torch.exp(new_log_probs - old_log_probs_stack),
                    1 - EPSILON,
                    1 + EPSILON,
                )
                * old_advantage_tensor,
            )
        )
        policy_loss.backward()
        policy_optimiser.step()
        del kv_cache
for i in range(EPISODE_NUM // 50):
    mean_rewards.append(np.mean(episode_rewards[i * 50 : (i + 1) * 50]))
print(mean_rewards)

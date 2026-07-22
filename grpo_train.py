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
GROUP_SIZE = 4
BATCH_SIZE = 1
BOTTNECK_RANK = 16
LORA_ALPHA = BOTTNECK_RANK * 2
NUM_STEPS = 15000
MAX_LR = 1e-4
MIN_LR = 1e-5
KL_CONSTANT = 0.01  # very low but i wanna see what my model looks like as it expeditions out of the trust region, also sft model is very stupid compared to SOTA so gotta use a smaller number than SOTA to allow the model to change more
CHECKPOINT = 2
device = "cuda" if torch.cuda.is_available() else "cpu"
cwd = os.getcwd()
data = load_from_disk("processed-metamathqa")
data = data.filter(lambda x: len(x["input_ids"]) <= MAX_TOKENS)
data = data.train_test_split(test_size=0.1, train_size=0.9)  # type: ignore
train_data = data["train"]
val_data = data["test"]
tokeniser = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
new_policy_v0 = load_cooked_model(
    BOTTNECK_RANK,
    LORA_ALPHA,
    params_path=cwd + f"/Atto-Math-SFT-V0-checkpoint{CHECKPOINT}.pt",
)
new_policy_v0.enable_input_require_grads()
new_policy_v0.gradient_checkpointing_enable()
old_policy_v0 = load_cooked_model(
    BOTTNECK_RANK,
    LORA_ALPHA,
    params_path=cwd + f"/Atto-Math-SFT-V0-checkpoint{CHECKPOINT}.pt",
)
old_policy_v0.config.use_cache = False
original_policy_v0 = load_cooked_model(
    BOTTNECK_RANK,
    LORA_ALPHA,
    params_path=cwd + f"/Atto-Math-SFT-V0-checkpoint{CHECKPOINT}.pt",
)
train_dataloader = Dataloader(train_data, True, tokeniser, BATCH_SIZE)
val_dataloader = Dataloader(val_data, False, tokeniser, BATCH_SIZE)
train_iter = iter(train_dataloader)
val_iter = iter(val_dataloader)
policy_optimiser = bnb.optim.PagedAdamW8bit(  # type: ignore
    params=[param for param in new_policy_v0.parameters() if param.requires_grad],
    lr=MAX_LR,
)
original_policy_v0.to(device)  # type: ignore
old_policy_v0.to(device)  # type: ignore
new_policy_v0.to(device)  # type: ignore

old_adaptor_params = [
    param.data
    for name, param in old_policy_v0.named_parameters()
    if ".adaptor." in name
]
new_adaptor_params = [
    param.data
    for name, param in new_policy_v0.named_parameters()
    if ".adaptor." in name
]


def quantise_test(model):
    return model.model.layers[0].self_attn.q_proj.original_layer.weight.bnb_quantized


# %%
# grpo time fellas ;D

EPISODE_NUM = 1000
EPSILON = 0.2
OLD_POLICY_LOOPS = 4
NEW_POLICY_LOOPS = 8
for param in original_policy_v0.parameters():
    param.requires_grad = False
episode_rewards = []
mean_rewards = []
for episode in range(EPISODE_NUM):
    # if episode % 100 == 0:
    #     print(episode)
    print(episode)
    print(quantise_test(original_policy_v0))
    print(quantise_test(old_policy_v0))
    print(quantise_test(new_policy_v0))
    torch._foreach_copy_(old_adaptor_params, new_adaptor_params)
    for param in old_policy_v0.parameters():
        param.requires_grad = False
    old_log_probs_stack = []
    old_returns_stack = []
    tokenised_prompt_stack = []
    original_tokenised_prompt_stack = []
    combined_mask_stack = []
    with torch.no_grad():
        for i in range(OLD_POLICY_LOOPS):
            print(i)
            original_param_dict = next(train_iter)
            current_batch = original_param_dict["input_ids"].shape[0]
            mask = (
                original_param_dict["attention_mask"]
                .repeat_interleave(GROUP_SIZE, dim=0)
                .to(device)
            )
            input = (
                original_param_dict["input_ids"]
                .repeat_interleave(GROUP_SIZE, dim=0)
                .to(device),
                mask,
            )
            tokenised_prompt = (
                original_param_dict["input_ids"]
                .repeat_interleave(GROUP_SIZE, dim=0)
                .to(device)
            )
            original_tokenised_prompt_stack.append(tokenised_prompt)
            finished = torch.zeros(
                current_batch * GROUP_SIZE,
                dtype=torch.bool,
            ).to(device)
            finished_idx = torch.full(
                (current_batch * GROUP_SIZE,), -1, dtype=torch.long
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
                        torch.ones(GROUP_SIZE * current_batch, 1, device=device),
                    ),
                    dim=-1,
                )
                input = (next_word.unsqueeze(1), mask)
                just_finished = (~finished) & (
                    (next_word == tokeniser.eos_token_id) | (next_word == imend_token)
                )
                finished = finished | just_finished
                tokenised_prompt = torch.cat(
                    (tokenised_prompt, torch.unsqueeze(next_word, dim=1)),
                    dim=-1,
                )
                finished_idx = torch.where(
                    just_finished,
                    torch.full_like(finished_idx, tokenised_prompt.shape[-1]),
                    finished_idx,
                )
            finished_idx = torch.where(
                finished_idx == -1,
                torch.full_like(finished_idx, tokenised_prompt.shape[-1]),
                finished_idx,
            )
            decoded_out = tokeniser.batch_decode(tokenised_prompt)
            old_log_probs_tensor = torch.stack(old_log_probs_lst)
            old_log_probs_stack.append(old_log_probs_tensor)
            for i in range(current_batch):
                group_correct = 0
                maj_dict = {}
                for row in decoded_out[i * GROUP_SIZE : i * GROUP_SIZE + GROUP_SIZE]:
                    try:
                        if float(extract_answer(row).replace(",", "")) == float(
                            original_param_dict["labels"][i].replace(",", "")  # type: ignore
                        ):
                            old_returns_tensor = torch.ones_like(
                                old_log_probs_tensor, device=device
                            )
                            group_correct += 1
                        else:
                            old_returns_tensor = torch.zeros_like(
                                old_log_probs_tensor, device=device
                            )
                    except ValueError:
                        print("Value error oh no")
                        old_returns_tensor = torch.zeros_like(
                            old_log_probs_tensor, device=device
                        )
                episode_rewards.append(group_correct)
            original_prompt_idx = original_tokenised_prompt_stack[-1].shape[-1]
            seq_len = tokenised_prompt.shape[-1]
            positions = torch.arange(1, seq_len, device=device).unsqueeze(0)
            combined_mask = (
                (positions >= original_prompt_idx)
                & (positions <= finished_idx.unsqueeze(1))
            ).float()
            combined_mask_stack.append(combined_mask)
            tokenised_prompt_stack.append(tokenised_prompt)
            old_returns_stack.append(old_returns_tensor)  # type: ignore
            del out, original_param_dict, kv_cache  # type: ignore
    old_advantage_tensor = torch.stack(old_returns_stack) - torch.mean(
        torch.stack(old_returns_stack), dim=-1
    )  # type: ignore
    old_log_probs_stack = torch.stack(old_log_probs_stack)
    tokenised_prompt_stack = torch.stack(tokenised_prompt_stack)
    combined_mask_stack = torch.stack(combined_mask_stack)
    original_tokenised_prompt_stack = torch.stack(original_tokenised_prompt_stack)
    for i in range(NEW_POLICY_LOOPS):
        out = new_policy_v0(input_ids=tokenised_prompt_stack)
        logits = out.logits
        log_probs = F.log_softmax(logits, dim=-1)
        targets = tokenised_prompt_stack[:, :, 1:]
        new_log_probs = log_probs.gather(-1, targets)
        original_out = original_policy_v0(input_ids=tokenised_prompt_stack)
        original_logits = original_out.logits
        original_log_probs = F.log_softmax(original_logits, dim=-1)
        original_log_probs = original_log_probs.gather(-1, targets)
        policy_optimiser.zero_grad()
        old_advantage_tensor *= combined_mask_stack  # breaks computation graph btw but its ok cos we wont backprop thru to old model
        policy_loss = (
            -(
                (
                    torch.minimum(
                        torch.exp(new_log_probs - old_log_probs_stack)
                        * old_advantage_tensor,
                        torch.clip(
                            torch.exp(new_log_probs - old_log_probs_stack),
                            1 - EPSILON,
                            1 + EPSILON,
                        )
                        * old_advantage_tensor,
                    ).sum()
                )
                - (
                    (
                        (
                            torch.exp(original_log_probs - new_log_probs)
                            - original_log_probs
                            + new_log_probs
                            - 1
                        )
                        * combined_mask_stack
                    ).sum()
                    * KL_CONSTANT
                )
            )
            / combined_mask_stack.sum()
        )
        policy_loss.backward()
        policy_optimiser.step()
for i in range(EPISODE_NUM // 50):
    mean_rewards.append(np.mean(episode_rewards[i * 50 : (i + 1) * 50]))
print(mean_rewards)
# KL constant very low so gotta do some actual model generation here once in a while to see if the model starts going insane but still gives high rewards like for example language mixing like in deepseek zero

# %%
# testing ground
print(quantise_test(original_policy_v0))
print(quantise_test(old_policy_v0))
print(quantise_test(new_policy_v0))

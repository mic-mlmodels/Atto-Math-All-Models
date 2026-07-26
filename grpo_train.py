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
BATCH_SIZE = 16
BOTTNECK_RANK = 16
LORA_ALPHA = BOTTNECK_RANK * 2
NUM_STEPS = 15000
MAX_LR = 1e-4
MIN_LR = 1e-5
KL_CONSTANT = 0.01  # very low but i wanna see what my model looks like as it expeditions out of the trust region, also sft model is very stupid compared to SOTA so gotta use a smaller number than SOTA to allow the model to change more
CHECKPOINT = 4
NEW_POLICY_SLICE_SIZE = 1  # seems small but can NOT raise this any higher even on my 5070 due to the output vocab size being so large
EPSILON = 0.2
OLD_POLICY_LOOPS = 4
NEW_POLICY_LOOPS = 8
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
for param in original_policy_v0.parameters():
    param.requires_grad = False
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

EPISODE_NUM = 10
for param in original_policy_v0.parameters():
    param.requires_grad = False
mean_rewards = []
episode_rewards = []
val_episode_rewards = []
val_mean_rewards = []
for episode in range(EPISODE_NUM):
    if episode % 10 == 0:
        val_episode_corrects = 0
        with torch.no_grad():
            for i in range(OLD_POLICY_LOOPS):
                print(i)
                try:
                    original_param_dict = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_dataloader)
                    original_param_dict = next(train_iter)
                current_batch = original_param_dict["input_ids"].shape[0]
                labels = original_param_dict["labels"]
                query_length = (labels != -100).long().argmax(dim=-1)
                highest_query_length = max(query_length)
                query_ids = torch.stack(
                    [
                        F.pad(
                            t,
                            (int(highest_query_length) - int(query_length[i]), 0),
                            value=tokeniser.pad_token_id,
                        )[:highest_query_length]
                        for i, t in enumerate(original_param_dict["input_ids"])
                    ]
                )
                query_attention_mask = (query_ids != tokeniser.pad_token_id).long()
                mask = query_attention_mask.repeat_interleave(GROUP_SIZE, dim=0).to(
                    device
                )
                input = (
                    query_ids.repeat_interleave(GROUP_SIZE, dim=0).to(device),
                    mask,
                )
                tokenised_prompt = query_ids.repeat_interleave(GROUP_SIZE, dim=0).to(
                    device
                )
                finished = torch.zeros(
                    current_batch * GROUP_SIZE,
                    dtype=torch.bool,
                ).to(device)
                finished_idx = torch.full(
                    (current_batch * GROUP_SIZE,), -1, dtype=torch.long
                ).to(device)
                imend_token = tokeniser.convert_tokens_to_ids("<|im_end|>")
                kv_cache = None
                while not finished.all() and tokenised_prompt.shape[-1] < 1024:
                    out = new_policy_v0(
                        *input,
                        past_key_values=kv_cache,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                    kv_cache = out.past_key_values
                    logits = out.logits
                    probs = F.softmax(logits[:, -1, :], dim=-1)
                    dist_obj = torch.distributions.Categorical(probs)
                    next_word = dist_obj.sample()
                    mask = torch.cat(
                        (
                            mask,
                            torch.ones(GROUP_SIZE * current_batch, 1, device=device),
                        ),
                        dim=-1,
                    )
                    input = (next_word.unsqueeze(1), mask)
                    just_finished = (~finished) & (
                        (next_word == tokeniser.eos_token_id)
                        | (next_word == imend_token)
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
                for i in range(current_batch):
                    group_correct = 0
                    for j, row in enumerate(
                        decoded_out[i * GROUP_SIZE : i * GROUP_SIZE + GROUP_SIZE]
                    ):
                        if float(extract_answer(row).replace(",", "")) == float(
                            extract_answer(
                                tokeniser.decode(
                                    original_param_dict["input_ids"][i]
                                ).replace(",", "")
                            )  # type: ignore
                        ):
                            group_correct += 1
                    val_episode_corrects += group_correct
                del out, original_param_dict, kv_cache  # type: ignore
            val_episode_rewards.append(val_episode_corrects)

    print(f"episode: {episode}")
    torch._foreach_copy_(old_adaptor_params, new_adaptor_params)  # type: ignore
    for param in old_policy_v0.parameters():
        param.requires_grad = False
    old_log_probs_stack = []
    old_returns_stack = []
    tokenised_prompt_stack = []
    original_tokenised_prompt_stack = []
    combined_mask_stack = []
    attention_mask_stack = []
    episode_corrects = 0
    with torch.no_grad():
        for i in range(OLD_POLICY_LOOPS):
            print(i)
            try:
                original_param_dict = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                original_param_dict = next(train_iter)
            current_batch = original_param_dict["input_ids"].shape[0]
            labels = original_param_dict["labels"]
            query_length = (labels != -100).long().argmax(dim=-1)
            highest_query_length = max(query_length)
            query_ids = torch.stack(
                [
                    F.pad(
                        t,
                        (int(highest_query_length) - int(query_length[i]), 0),
                        value=tokeniser.pad_token_id,
                    )[:highest_query_length]
                    for i, t in enumerate(original_param_dict["input_ids"])
                ]
            )
            query_attention_mask = (query_ids != tokeniser.pad_token_id).long()
            mask = query_attention_mask.repeat_interleave(GROUP_SIZE, dim=0).to(device)
            input = (
                query_ids.repeat_interleave(GROUP_SIZE, dim=0).to(device),
                mask,
            )
            tokenised_prompt = query_ids.repeat_interleave(GROUP_SIZE, dim=0).to(device)
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
            old_log_probs_tensor = F.pad(
                old_log_probs_tensor, (0, 0, int(highest_query_length) - 1, 0)
            ).T
            old_log_probs_stack.append(old_log_probs_tensor)
            for i in range(current_batch):
                group_correct = 0
                for j, row in enumerate(
                    decoded_out[i * GROUP_SIZE : i * GROUP_SIZE + GROUP_SIZE]
                ):
                    try:
                        if (extract_answer(row).replace(",", "")) == extract_answer(
                            tokeniser.decode(
                                original_param_dict["input_ids"][i]
                            ).replace(",", "")
                        ):
                            old_returns_tensor = torch.ones_like(
                                old_log_probs_tensor[0, i * GROUP_SIZE + j],
                                device=device,
                            )
                            group_correct += 1
                        else:
                            old_returns_tensor = torch.zeros_like(
                                old_log_probs_tensor[0, i * GROUP_SIZE + j],
                                device=device,
                            )
                    except ValueError:
                        print("Value error oh no")
                        old_returns_tensor = torch.zeros_like(
                            old_log_probs_tensor[0, i * GROUP_SIZE + j],
                            device=device,
                        )
                    old_returns_stack.append(old_returns_tensor)  # type: ignore
                episode_corrects += group_correct
            original_prompt_idx = original_tokenised_prompt_stack[-1].shape[-1]
            seq_len = tokenised_prompt.shape[-1]
            positions = torch.arange(1, seq_len, device=device).unsqueeze(0)
            combined_mask = (
                (positions >= original_prompt_idx)
                & (positions < finished_idx.unsqueeze(1))
            ).float()
            combined_mask_stack.append(combined_mask)
            tokenised_prompt_stack.append(tokenised_prompt)
            attention_mask_stack.append(mask)
            del out, original_param_dict, kv_cache  # type: ignore
        episode_rewards.append(episode_corrects)

    max_sequence_length = max(t.shape[-1] for t in tokenised_prompt_stack)
    old_log_probs_stack = torch.cat(
        [
            F.pad(t, (0, (max_sequence_length - 1) - t.shape[-1], 0, 0))
            for t in old_log_probs_stack
        ],
        dim=0,
    )
    tokenised_prompt_stack = torch.cat(
        [
            F.pad(t, (0, max_sequence_length - t.shape[-1], 0, 0))
            for t in tokenised_prompt_stack
        ],
        dim=0,
    )
    combined_mask_stack = torch.cat(
        [
            F.pad(t, (0, (max_sequence_length - 1) - t.shape[-1], 0, 0))
            for t in combined_mask_stack
        ],
        dim=0,
    )
    attention_mask_stack = torch.cat(
        [
            F.pad(t, (0, max_sequence_length - t.shape[-1], 0, 0))
            for t in attention_mask_stack
        ],
        dim=0,
    )
    old_returns_stack = torch.stack(old_returns_stack).view(-1, GROUP_SIZE)  # type: ignore
    old_advantage_stack = old_returns_stack - old_returns_stack.mean(
        dim=-1, keepdim=True
    )
    old_advantage_stack = old_advantage_stack.view(-1, 1)
    for i in range(NEW_POLICY_LOOPS):
        policy_optimiser.zero_grad()
        for i in range(0, tokenised_prompt_stack.shape[0], NEW_POLICY_SLICE_SIZE):
            sliced_tokenised_prompt_stack = tokenised_prompt_stack[
                i : i + NEW_POLICY_SLICE_SIZE
            ]
            sliced_old_log_probs = old_log_probs_stack[i : i + NEW_POLICY_SLICE_SIZE]
            sliced_old_advantage_tensor = old_advantage_stack[
                i : i + NEW_POLICY_SLICE_SIZE
            ]
            sliced_combined_mask_stack = combined_mask_stack[
                i : i + NEW_POLICY_SLICE_SIZE
            ]
            sliced_attention_mask_stack = attention_mask_stack[
                i : i + NEW_POLICY_SLICE_SIZE
            ]
            out = new_policy_v0(
                input_ids=sliced_tokenised_prompt_stack,
                attention_mask=sliced_attention_mask_stack,
            )
            logits = out.logits
            log_probs = F.log_softmax(logits, dim=-1)
            del out, logits
            targets = sliced_tokenised_prompt_stack[:, 1:]
            new_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            original_out = original_policy_v0(
                input_ids=sliced_tokenised_prompt_stack,
                attention_mask=sliced_attention_mask_stack,
            )
            original_logits = original_out.logits
            original_log_probs = F.log_softmax(original_logits, dim=-1)
            original_log_probs = original_log_probs.gather(
                -1, targets.unsqueeze(-1)
            ).squeeze(-1)
            del original_out, original_logits
            sliced_old_advantage_tensor = (
                sliced_old_advantage_tensor * sliced_combined_mask_stack
            )
            policy_loss = (
                -(
                    (
                        torch.minimum(
                            torch.exp(new_log_probs - sliced_old_log_probs)
                            * sliced_old_advantage_tensor,
                            torch.clip(
                                torch.exp(new_log_probs - sliced_old_log_probs),
                                1 - EPSILON,
                                1 + EPSILON,
                            )
                            * sliced_old_advantage_tensor,
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
                            * sliced_combined_mask_stack
                        ).sum()
                        * KL_CONSTANT
                    )
                )
                / combined_mask_stack.sum()
            )
            policy_loss.backward()
            del (
                sliced_combined_mask_stack,
                sliced_attention_mask_stack,
                sliced_old_advantage_tensor,
                sliced_old_log_probs,
                sliced_tokenised_prompt_stack,
                log_probs,
                targets,
            )
        policy_optimiser.step()
for i in range(EPISODE_NUM // 5):
    mean_rewards.append(np.mean(episode_rewards[i * 5 : (i + 1) * 5]))
print(mean_rewards)
for i in range(EPISODE_NUM // (5 * 10)):
    val_mean_rewards.append(np.mean(val_episode_rewards[i * 5 : (i + 1) * 5]))
print(val_mean_rewards)
# KL constant very low so gotta do some actual model generation here once in a while to see if the model starts going insane but still gives high rewards like for example language mixing like in deepseek zero

# %%
# testing ground
print(quantise_test(original_policy_v0))
print(quantise_test(old_policy_v0))
print(quantise_test(new_policy_v0))

tokeniser.decode(train_data[0])

with torch.no_grad():
    tiny_out = new_policy_v0(
        input_ids=torch.zeros(1, 2, dtype=torch.long, device=device)
    )
    print(tiny_out.logits.dtype)

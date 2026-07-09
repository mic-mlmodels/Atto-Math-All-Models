# %%
# imports
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR
import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
import bitsandbytes as bnb
from dataloader import Dataloader
from qlora import adapt_model
import torch.nn.functional as F

# %%
# constants

MAX_TOKENS = 768
BATCH_SIZE = 2
BOTTNECK_RANK = 16
LORA_ALPHA = BOTTNECK_RANK * 2
NUM_STEPS = 15000
MAX_LR = 1e-4
MIN_LR = 1e-5

# %%
# data load
data = load_from_disk("processed-metamathqa")
data = data.filter(
    lambda x: len(x["input_ids"]) <= MAX_TOKENS
)  # maybe should have put this in the initial data processing setup but this keeps it more flexible if i need to change filtering for grpo for example
data = data.train_test_split(test_size=0.1, train_size=0.9)  # type: ignore
train_data = data["train"]
val_data = data["test"]

# %%
# base model
device = "cuda" if torch.cuda.is_available() else "cpu"
tokeniser = AutoTokenizer.from_pretrained("Qwen2.5-1.5B base model")
model = AutoModelForCausalLM.from_pretrained("Qwen2.5-1.5B base model")
for param in model.parameters():
    param.requires_grad = False
model.config.use_cache = False
model.enable_input_require_grads()
model.gradient_checkpointing_enable()
train_dataloader = Dataloader(train_data, True, tokeniser, BATCH_SIZE)
val_dataloader = Dataloader(val_data, False, tokeniser, BATCH_SIZE)
train_iter = iter(train_dataloader)
val_iter = iter(val_dataloader)
adapt_model(model, BOTTNECK_RANK, device, LORA_ALPHA)
model.to(device)  # type: ignore
optimiser = bnb.optim.PagedAdamW8bit(  # type: ignore
    params=[param for param in model.parameters() if param.requires_grad],
    lr=MAX_LR,
)

# %%
# lr scheduler
NUM_UPDATES = NUM_STEPS // 8
WARMUP_UPDATES = NUM_UPDATES // 10

warmup_scheduler = LinearLR(
    optimiser, start_factor=0.05, end_factor=1.0, total_iters=WARMUP_UPDATES
)

decay_scheduler = CosineAnnealingLR(
    optimiser, T_max=(NUM_UPDATES - WARMUP_UPDATES), eta_min=MIN_LR
)

scheduler = ChainedScheduler([warmup_scheduler, decay_scheduler])
# %%
# training yippee :D

len(train_data[0]["input_ids"])
len(train_data[0]["attention_mask"])
len(train_data[0]["labels"])
len(train_data)
mean_train_loss_lst = []
mean_val_loss_lst = []

# %%
# training yippee :D
print("Atto-Math-SFT model cooking...")
train_loss_lst = []
val_loss_lst = []
for step in range(NUM_STEPS):
    model.train()
    try:
        param_dict = next(train_iter)
    except StopIteration:
        train_iter = iter(train_dataloader)
        param_dict = next(train_iter)
    param_dict = {k: v.to(device) for k, v in param_dict.items()}
    out = model(**param_dict)
    loss = out.loss / 8
    loss.backward()
    train_loss_lst.append(loss.item() * 8)
    if step % 8 == 0:
        norm = torch.nn.utils.clip_grad_norm_(
            [param for param in model.parameters() if param.requires_grad], max_norm=1.0
        )
        optimiser.step()
        scheduler.step()
        optimiser.zero_grad()
        train_loss_mean = np.mean(train_loss_lst)
        mean_train_loss_lst.append(train_loss_mean)
        train_loss_lst = []
        model.eval()
        with torch.no_grad():
            try:
                val_param_dict = next(val_iter)
            except StopIteration:
                val_iter = iter(val_dataloader)
                val_param_dict = next(val_iter)
            val_param_dict = {k: v.to(device) for k, v in val_param_dict.items()}
            val_out = model(**val_param_dict)
            val_loss = val_out.loss
            val_loss_lst.append(val_loss.item())
        print(f"step: {step}, train loss: {train_loss_mean}, grad norm: {norm}")
        del val_loss, val_out, val_param_dict
    if step % 64 == 0:
        val_loss_mean = np.mean(val_loss_lst)
        mean_val_loss_lst.append(val_loss_mean)
        val_loss_lst = []
        print(f"val loss: {val_loss_mean}")

    del out, loss, param_dict
print("Atto-Math-SFT model cooked!")

torch.save(
    {
        "model_state_dict": {
            name: param
            for name, param in model.named_parameters()
            if param.requires_grad
        },
        "mean_train_loss_lst": mean_train_loss_lst,
        "mean_val_loss_lst": mean_val_loss_lst,
    },
    "Atto-Math-SFT-V0-checkpoint1.pt",
)

# %%
# eval time

model.eval()
test_prompt = tokeniser.apply_chat_template(
    [
        {
            "role": "system",
            "content": "You are a helpful assistant. You must think step-by-step inside <think> tags before providing the final answer after ####.",
        },
        {
            "role": "user",
            "content": "Given the complex expression Z = (3 + 2i)(1 - 4i) / (2 + i), simplify Z completely into its standard rectangular form a + bi.",
        },
    ],
    tokenize=False,
    add_generation_prompt=True,
)
test_prompt = test_prompt
with torch.no_grad():
    tokenised_prompt = torch.unsqueeze(
        torch.tensor(tokeniser(test_prompt)["input_ids"]).to(device), dim=0
    )
    next_word = 0
    imend_token = tokeniser.convert_tokens_to_ids("<|im_end|>")
    while (
        next_word != tokeniser.eos_token_id
        and tokenised_prompt.shape[-1] < 1024
        and next_word != imend_token
    ):
        out = model(tokenised_prompt)
        logits = out.logits
        probs = F.softmax(logits[:, -1, :], dim=-1)
        dist_obj = torch.distributions.Categorical(probs)
        next_word = dist_obj.sample()
        tokenised_prompt = torch.cat(
            (tokenised_prompt, torch.unsqueeze(next_word, dim=0)),
            dim=-1,
        )
    print(tokeniser.decode(torch.squeeze(tokenised_prompt)))

# %%
# eval time but hf library

model.eval()
model.config.use_cache = True
imend_token = tokeniser.convert_tokens_to_ids("<|im_end|>")
inputs = tokeniser(test_prompt, return_tensors="pt").to(device)
with torch.no_grad():
    generated = model.generate(  # type: ignore
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
        eos_token_id=[tokeniser.eos_token_id, imend_token],
    )
print(tokeniser.decode(generated[0]))

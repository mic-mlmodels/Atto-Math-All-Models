# %%
# imports
import numpy as np
import torch
from datasets import load_from_disk, Dataset
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import bitsandbytes as bnb
from dataloader import Dataloader
from qlora import adapt_model

data = load_from_disk("processed-metamathqa")

data = data.train_test_split(test_size=0.1, train_size=0.9)  # type: ignore
train_data = data["train"]
val_data = data["test"]

# %%
# base model
BATCH_SIZE = 2
BOTTNECK_RANK = 2
LORA_ALPHA = BOTTNECK_RANK * 2
LR = 3e-4
NUM_STEPS = 1000
device = "cuda" if torch.cuda.is_available() else "cpu"
tokeniser = AutoTokenizer.from_pretrained("Qwen2.5-1.5B base model")
model = AutoModelForCausalLM.from_pretrained("Qwen2.5-1.5B base model")
train_dataloader = Dataloader(train_data, True, tokeniser, BATCH_SIZE)
val_dataloader = Dataloader(val_data, False, tokeniser, BATCH_SIZE)
train_iter = iter(train_dataloader)
val_iter = iter(val_dataloader)
adapt_model(model, BOTTNECK_RANK, device, LORA_ALPHA)
model.to(device)  # type: ignore
optimiser = bnb.optim.PagedAdamW8bit(
    params=[param for param in model.parameters() if param.requires_grad], lr=LR
)
# %%
# training yippee :D

len(train_data[0]["input_ids"])
len(train_data[0]["attention_mask"])
len(train_data[0]["labels"])
len(train_data)

# %%
# training yippee :D
print("Atto-Math-SFT model cooking...")
train_loss_lst = []
mean_train_loss_lst = []
val_loss_lst = []
mean_val_loss_lst = []
for step in range(NUM_STEPS):
    model.train()
    try:
        param_dict = next(train_iter)
    except StopIteration:
        train_iter = iter(train_dataloader)
        param_dict = next(train_iter)
    param_dict = {k: v.to(device) for k, v in param_dict.items()}
    out = model(**param_dict)
    loss = out.loss
    loss.backward()
    optimiser.step()
    optimiser.zero_grad()
    train_loss_lst.append(loss.item())
    if step % 20 == 0:
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
        print(f"train loss: {train_loss_mean}")
        del val_loss, val_out, val_param_dict
    if step % 100 == 0:
        val_loss_mean = np.mean(val_loss_lst)
        mean_val_loss_lst.append(val_loss_mean)
        val_loss_lst = []
        print(f"val loss: {val_loss_mean}")

    del out, loss, param_dict
print("Atto-Math-SFT model cooked!")

# %%
# eval time

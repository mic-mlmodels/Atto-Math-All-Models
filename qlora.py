from torch import bfloat16, uint8
import torch.nn as nn
import bitsandbytes as bnb
from bitsandbytes.nn.modules import Linear4bit, Params4bit


class LayerAdaptor(nn.Module):
    def __init__(self, original_layer, bottleneck_rank, device, lora_alpha):
        super().__init__()
        self.alpha = lora_alpha
        self.bottleneck_rank = bottleneck_rank
        self.original_layer = Linear4bit(
            original_layer.in_features,
            original_layer.out_features,
            bias=False,
            compute_dtype=bfloat16,
            quant_type="nf4",
            quant_storage=uint8,
        )
        self.original_layer.weight = Params4bit(
            original_layer.weight.data.clone().cpu(),
            requires_grad=False,
        )
        for param in self.original_layer.parameters():
            param.requires_grad = False
            self.adaptor = nn.Sequential(
                nn.Linear(
                    self.original_layer.in_features,
                    self.bottleneck_rank,
                    dtype=bfloat16,
                ),
                nn.Linear(
                    self.bottleneck_rank,
                    self.original_layer.out_features,
                    dtype=bfloat16,
                ),
            )
            nn.init.kaiming_uniform_(self.adaptor[0].weight, a=1)  # type: ignore
            nn.init.zeros_(self.adaptor[1].weight)  # type: ignore

    def forward(self, x):
        return (
            self.original_layer(x) + self.adaptor(x) * self.bottleneck_rank / self.alpha
        )


def adapt_model(model, bottleneck_rank, device, lora_alpha):
    adapt_todo_lst = []
    for layer_name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            adapt_todo_lst.append((layer_name, layer))
    for adapt_todo_name, adapt_todo_layer in adapt_todo_lst:
        modules = adapt_todo_name.split(".")
        parent = model
        for module in modules[:-1]:
            if module.isdigit():
                parent = parent[int(module)]
            else:
                parent = getattr(parent, module)
        setattr(
            parent,
            modules[-1],
            LayerAdaptor(adapt_todo_layer, bottleneck_rank, device, lora_alpha),
        )

from torch import bfloat16, uint8
import torch.nn as nn
import bitsandbytes as bnb
from bitsandbytes.nn.modules import Linear4bit, Params4bit


class LayerAdaptor(nn.Module):
    def __init__(self, original_layer, bottleneck_rank, layer_type, device, lora_alpha):
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
        ).cuda()
        for param in self.original_layer.parameters():
            param.requires_grad = False
        if layer_type == "linear":
            self.adaptor = nn.Sequential(
                nn.Linear(self.original_layer.in_features, self.bottleneck_rank),
                nn.Linear(
                    self.bottleneck_rank,
                    self.original_layer.out_features,
                ),
            )
            nn.init.kaiming_uniform_(self.adaptor[0].weight, a=1)  # type: ignore
            nn.init.zeros_(self.adaptor[1].weight)  # type: ignore

    def forward(self, x):
        return (
            self.original_layer(x) + self.adaptor(x) * self.bottleneck_rank / self.alpha
        )


def adapt_model(model, bottleneck_rank, device, lora_alpha):
    for layer_name, layer in model.named_children():
        if layer_name == "lm_head":
            continue
        if isinstance(layer, nn.Linear):
            setattr(
                model,
                layer_name,
                LayerAdaptor(layer, bottleneck_rank, "linear", device, lora_alpha),
            )
        else:
            adapt_model(layer, bottleneck_rank, device, lora_alpha)

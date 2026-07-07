from torch import bfloat16, uint8
import torch.nn as nn
from transformers import Conv1D
import bitsandbytes as bnb
from bitsandbytes.nn.modules import Linear4bit


class LayerAdaptor(nn.Module):
    def __init__(self, original_layer, bottleneck_rank, layer_type):
        super().__init__()
        self.original_layer = Linear4bit(
            original_layer.in_features,
            original_layer.out_features,
            bias=False,
            compute_dtype=bfloat16,
            quant_type="nf4",
            quant_storage=uint8,
        )
        for param in self.original_layer.parameters():
            param.requires_grad = False
        if layer_type == "linear":
            self.adaptor = nn.Sequential(
                nn.Linear(self.original_layer.in_features, bottleneck_rank),
                nn.Linear(bottleneck_rank, self.original_layer.out_features),
            )

    def forward(self, x):
        return self.original_layer(x) + self.adaptor(x)


def adapt_model(model, bottleneck_rank):
    for layer_name, layer in model.named_children():
        if layer_name == "lm_head":
            continue
        if isinstance(layer, nn.Linear):
            setattr(model, layer_name, LayerAdaptor(layer, bottleneck_rank, "linear"))
        else:
            adapt_model(layer, bottleneck_rank)

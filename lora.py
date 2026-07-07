import torch.nn as nn
from transformers import Conv1D


class LayerAdaptor(nn.Module):
    def __init__(self, original_layer, bottleneck_rank, layer_type):
        super().__init__()
        self.original_layer = original_layer
        if layer_type == "linear":
            self.adaptor = nn.Sequential(
                nn.Linear(self.original_layer.in_features, bottleneck_rank),
                nn.Linear(bottleneck_rank, self.original_layer.out_features),
            )
        elif layer_type == "conv":
            self.adaptor = nn.Sequential(
                nn.Linear(self.original_layer.weight.shape[0], bottleneck_rank),
                nn.Linear(bottleneck_rank, self.original_layer.weight.shape[1]),
            )

    def forward(self, x):
        return self.original_layer(x) + self.adaptor(x)


def adapt_model(model, bottleneck_rank):
    for layer_name, layer in model.named_children():
        if layer_name == "lm_head":
            continue
        if isinstance(layer, nn.Linear):
            setattr(model, layer_name, LayerAdaptor(layer, bottleneck_rank, "linear"))
        elif isinstance(layer, Conv1D):
            setattr(model, layer_name, LayerAdaptor(layer, bottleneck_rank, "conv"))
        else:
            adapt_model(layer, bottleneck_rank)

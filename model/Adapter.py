import torch
import torch.nn as nn

class ResidualAdapter(nn.Module):
    def __init__(self, input_size, target, reduction=4):
        super().__init__()
        hidden_dim = max(1, input_size // reduction)

        self.scale = nn.Parameter(torch.ones(1))  

        self.adapter = nn.Sequential(
            nn.Conv2d(input_size, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, target, kernel_size=1)
        )

        self.residual = nn.Conv2d(input_size, target, kernel_size=1)

    def forward(self, x):
        return self.residual(x) + self.scale * self.adapter(x)


class Adapter(nn.Module):
    def __init__(self, clip_model, target, reduction=4,input_dim=None):
        super(Adapter, self).__init__()
        input_sizes = clip_model.token_c
        if input_dim is not None:
            input_sizes = [input_dim,input_dim,input_dim]
        for i, input_size in enumerate(input_sizes):
            self.add_module(f"{i}_adapter", ResidualAdapter(input_size, target, reduction))

    def forward(self, tokens):
        vision_features = []
        for i, token in enumerate(tokens):
            adapter_layer = getattr(self, f"{i}_adapter")
            vision_feature = adapter_layer(token).contiguous().permute(0, 2, 3, 1)
            vision_feature = vision_feature / vision_feature.norm(dim=-1, keepdim=True)
            vision_features.append(vision_feature)
        return vision_features
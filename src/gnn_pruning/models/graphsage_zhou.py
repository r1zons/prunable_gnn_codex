"""GraphSAGE with independent normalized branches as in Zhou et al. (2021)."""

import torch
from torch import nn
from torch.nn import functional as F

from .base import BaseNodeClassifier


class ZhouBranch(nn.Module):
    """One affine projection with ReLU and per-node normalization."""

    def __init__(self, inputs, outputs):
        super().__init__()
        self.projection = nn.Linear(inputs, outputs, bias=False)
        nn.init.xavier_uniform_(self.projection.weight)
        self.bias = nn.Parameter(torch.zeros(outputs))
        self.offset = nn.Parameter(torch.zeros(outputs))
        self.scale = nn.Parameter(torch.ones(outputs))

    def normalize(self, projected):
        """Apply the branch bias, activation and population-variance normalization."""
        if not self.training and not torch.is_grad_enabled():
            projected.add_(self.bias).relu_()
            mean = projected.mean(dim=1, keepdim=True)
            inverse_sd = projected.var(dim=1, unbiased=False, keepdim=True).add_(1e-9).rsqrt_()
            return projected.sub_(mean).mul_(self.scale).mul_(inverse_sd).add_(self.offset)
        active = F.relu(projected + self.bias)
        mean = active.mean(dim=1, keepdim=True)
        variance = active.var(dim=1, unbiased=False, keepdim=True)
        return (active - mean) * self.scale * torch.rsqrt(variance + 1e-9) + self.offset


class ZhouLayer(nn.Module):
    """Concatenate self and neighbor branches fed by a common dropped-out input."""

    def __init__(self, inputs, widths, dropout):
        super().__init__()
        self.self_branch = ZhouBranch(inputs, widths[0])
        self.neighbor_branch = ZhouBranch(inputs, widths[1])
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, adj):
        features = self.dropout(features)
        own = features.new_empty((len(features), 0))
        if self.self_branch.projection.out_features:
            own = self.self_branch.normalize(self.self_branch.projection(features))
        neighbor = self.neighbor_branch
        if not neighbor.projection.out_features:
            return own
        if self.training:
            projected = neighbor.projection(torch.sparse.mm(adj, features))
        else:
            projected = torch.sparse.mm(adj, neighbor.projection(features))
        return torch.cat((own, neighbor.normalize(projected)), dim=1)


class ZhouGraphSAGE(BaseNodeClassifier):
    """Two two-branch layers and a linear classifier over unit-normalized embeddings."""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=.3,
                 branch_widths=None):
        super().__init__(dropout)
        if num_layers != 2 or type(hidden_channels) is not int or hidden_channels <= 0:
            raise ValueError("Zhou GraphSAGE requires two layers and a positive shared branch width")
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        widths = ((hidden_channels, hidden_channels),) * 2 if branch_widths is None else branch_widths
        if (len(widths) != 2 or any(len(pair) != 2 or not sum(pair) or
                                   any(type(w) is not int or not 0 <= w <= hidden_channels for w in pair)
                                   for pair in widths)):
            raise ValueError("Invalid Zhou branch widths")
        self.layers = nn.ModuleList([
            ZhouLayer(in_channels, widths[0], dropout),
            ZhouLayer(sum(widths[0]), widths[1], dropout),
        ])
        self.classifier = nn.Linear(sum(widths[1]), out_channels, bias=False)
        nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier_bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, data, adj=None):
        """Accept a resident feature tensor and sparse adjacency or a data object with x and adj."""
        features, adj = (data.x, data.adj) if adj is None else (data, adj)
        for layer in self.layers:
            features = layer(features, adj)
        if not self.training and not torch.is_grad_enabled():
            features = F.normalize(features, p=2, dim=1, out=features)
        else:
            features = F.normalize(features, p=2, dim=1)
        features = F.dropout(features, p=self.dropout, training=self.training)
        return self.classifier(features) + self.classifier_bias

    @property
    def branch_widths(self):
        """Current self/neighbor widths in concatenation order."""
        return tuple((layer.self_branch.projection.out_features,
                      layer.neighbor_branch.projection.out_features) for layer in self.layers)

    @property
    def hidden_channel_dims(self):
        return tuple(map(sum, self.branch_widths))

    def export_architecture_config(self):
        """Describe this variant for strict checkpoint reconstruction."""
        config = dict(name="graphsage_zhou", in_channels=self.in_channels,
                      hidden_channels=self.hidden_channels, out_channels=self.out_channels,
                      num_layers=2, dropout=self.dropout)
        if self.branch_widths != ((self.hidden_channels,) * 2,) * 2:
            config["branch_widths"] = [list(pair) for pair in self.branch_widths]
        return config

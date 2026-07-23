"""
MetaMMF-GCN model adapted from the official TOIS 2024 implementation.

Reference:
    Liu, Wei, Liu, Wang, Nie, and Chua (2023), "Dynamic Multimodal Fusion
    via Meta-Learning Towards Micro-Video Recommendation", ACM TOIS 42(2).

Official implementation:
    https://github.com/iLearn-Lab/TOIS24-MetaMMF

Fidelity notes:
    - The dynamic fusion network follows official Fusion_model.Net:
      concatenate visual/acoustic/textual features, predict a meta signal with
      an MLP, pass through four CPD layers, and concatenate an item ID
      embedding to produce a 64-dimensional item representation.
    - The graph backbone follows official GCN_model.GCN: two 64->64 GraphSAGE
      layers with LeakyReLU, dropout 0.5 after layer 1, and dropout 0.2 after
      layer 2, optimized with BPR loss.
    - The official code depends on PyTorch Geometric SAGEConv. This local
      version implements the equivalent mean-neighbor GraphSAGE operation with
      a torch sparse adjacency so it can run in the current environment.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MetaMMFConfig:
    rank: int = 16
    signal_dim: int = 5
    fusion_dims: tuple = (1024, 512, 256, 32)
    item_id_dim: int = 32
    sage_dim: int = 64
    dropout1: float = 0.5
    dropout2: float = 0.2


class CPDLayer(nn.Module):
    """Official CPDLayer with an algebraically equivalent memory-light forward."""

    def __init__(self, rank, signal_dim, in_dim, out_dim):
        super().__init__()
        self.rank = rank
        self.signal_dim = signal_dim
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.A = nn.Parameter(torch.empty(in_dim, rank))
        self.B = nn.Parameter(torch.empty(rank, out_dim))
        self.C = nn.Parameter(torch.empty(rank * signal_dim, 1))
        self.M_1 = nn.Parameter(torch.empty(in_dim, out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.B)
        nn.init.xavier_uniform_(self.C)
        nn.init.xavier_uniform_(self.M_1)

    def forward(self, signal, feature):
        # Equivalent to official:
        # AB = reshape(A @ B, (1, -1)); ABC = reshape(C @ AB, (R, s_dim, -1));
        # T_1 = sum(ABC, dim=0); w_1 = reshape(signal @ T_1, (-1, in_dim, out_dim)).
        ab = torch.matmul(self.A, self.B)
        c = self.C.view(self.rank, self.signal_dim)
        signal_scale = torch.matmul(signal, c.sum(dim=0))
        base = torch.matmul(feature, self.M_1)
        dynamic = torch.matmul(feature, ab) * signal_scale.unsqueeze(1)
        return base + dynamic


class MetaMMFFusion(nn.Module):
    def __init__(self, n_items, visual_dim, acoustic_dim, textual_dim, config):
        super().__init__()
        self.n_items = n_items
        self.config = config
        self.feature_dim = visual_dim + acoustic_dim + textual_dim

        self.mlp_signal = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, config.signal_dim),
        )
        self.item_E = nn.Parameter(torch.empty(n_items, config.item_id_dim))
        nn.init.xavier_uniform_(self.item_E)

        dims = [self.feature_dim] + list(config.fusion_dims)
        self.layers = nn.ModuleList([
            CPDLayer(config.rank, config.signal_dim, dims[idx], dims[idx + 1])
            for idx in range(len(dims) - 1)
        ])
        self.activation = nn.LeakyReLU()

    def forward(self, visual, acoustic, textual, item_ids):
        feature = torch.cat((visual, acoustic, textual), dim=1)
        signal = self.mlp_signal(feature)
        x = feature
        for layer in self.layers[:-1]:
            x = self.activation(layer(signal, x))
        x = self.layers[-1](signal, x)
        return torch.cat((x, self.item_E[item_ids]), dim=1)


class MeanSAGEConv(nn.Module):
    """PyG SAGEConv-style mean aggregation for a fixed sparse graph."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_neigh = nn.Linear(in_dim, out_dim)
        self.lin_self = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, mean_adj):
        neigh = torch.sparse.mm(mean_adj, x)
        return self.lin_neigh(neigh) + self.lin_self(x)


class MetaMMFGCN(nn.Module):
    def __init__(self, n_users, n_items, features, config=None, device=None):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.config = config or MetaMMFConfig()
        self.device = device or torch.device("cpu")

        visual = torch.as_tensor(features["visual"], dtype=torch.float32)
        acoustic = torch.as_tensor(features["acoustic"], dtype=torch.float32)
        textual = torch.as_tensor(features["textual"], dtype=torch.float32)
        self.register_buffer("visual_features", visual)
        self.register_buffer("acoustic_features", acoustic)
        self.register_buffer("textual_features", textual)

        self.user = nn.Parameter(torch.empty(n_users, self.config.sage_dim))
        nn.init.xavier_uniform_(self.user)
        self.fus = MetaMMFFusion(
            n_items,
            visual.shape[1],
            acoustic.shape[1],
            textual.shape[1],
            self.config,
        )
        item_out_dim = self.config.fusion_dims[-1] + self.config.item_id_dim
        if item_out_dim != self.config.sage_dim:
            raise ValueError(
                f"MetaMMF item output dim must equal sage_dim, got {item_out_dim} and {self.config.sage_dim}."
            )
        self.conv1 = MeanSAGEConv(self.config.sage_dim, self.config.sage_dim)
        self.conv2 = MeanSAGEConv(self.config.sage_dim, self.config.sage_dim)

    def fuse_all_items(self, fusion_batch_size=64):
        chunks = []
        for start in range(0, self.n_items, fusion_batch_size):
            end = min(start + fusion_batch_size, self.n_items)
            item_ids = torch.arange(start, end, device=self.visual_features.device)
            chunks.append(
                self.fus(
                    self.visual_features[start:end],
                    self.acoustic_features[start:end],
                    self.textual_features[start:end],
                    item_ids,
                )
            )
        return torch.cat(chunks, dim=0)

    def forward(self, mean_adj, fusion_batch_size=64):
        item_emb = self.fuse_all_items(fusion_batch_size=fusion_batch_size)
        x = torch.cat([self.user, item_emb], dim=0)
        x = self.conv1(x, mean_adj)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.config.dropout1, training=self.training)
        x = self.conv2(x, mean_adj)
        x = F.dropout(x, p=self.config.dropout2, training=self.training)
        return x[:self.n_users], x[self.n_users:self.n_users + self.n_items]

    def bpr_loss(self, users, pos_items, neg_items, mean_adj, fusion_batch_size=64):
        all_users, all_items = self.forward(mean_adj, fusion_batch_size=fusion_batch_size)
        user_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        pos_scores = torch.sum(user_emb * pos_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_emb, dim=1)
        return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))


def normalize_rows(matrix):
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)

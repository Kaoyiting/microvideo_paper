"""
LATTICE-style multimodal recommendation model.

Reference:
    Zhang et al. (2021), "Mining Latent Structures for Multimedia
    Recommendation", ACM MM 2021.

Official implementation:
    https://github.com/CRIPAC-DIG/LATTICE

Project adaptation:
    - Keeps the official high-level design: LightGCN collaborative backbone,
      modality-aware item-item kNN graphs, learnable modality fusion weights,
      item graph convolution, and BPR training.
    - Generalizes the official image/text inputs to the local feature sets used
      in this thesis project, such as visual/acoustic/textual or content
      embeddings.
    - Uses dense item-item graph construction because the local ShortVideo and
      MicroLens-100K item counts are small enough for thesis-scale baseline
      experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_similarity(features):
    features = F.normalize(features, p=2, dim=-1)
    return torch.matmul(features, features.t())


def build_knn_graph(similarity, topk):
    k = min(topk, similarity.size(1))
    values, indices = torch.topk(similarity, k=k, dim=-1)
    graph = torch.zeros_like(similarity)
    graph.scatter_(dim=-1, index=indices, src=values)
    return graph


def normalize_dense_adj(adj):
    rowsum = torch.sum(adj, dim=-1)
    d_inv_sqrt = torch.pow(rowsum + 1e-8, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    return d_inv_sqrt.unsqueeze(1) * adj * d_inv_sqrt.unsqueeze(0)


class LATTICE(nn.Module):
    """LATTICE-style model with a LightGCN collaborative backbone."""

    def __init__(
        self,
        dataset,
        modality_features,
        embedding_dim=64,
        feat_embed_dim=64,
        ui_layers=2,
        item_layers=1,
        topk=10,
        lambda_coeff=0.9,
        device="cpu",
    ):
        super().__init__()
        if not modality_features:
            raise ValueError("LATTICE requires at least one modality feature matrix.")

        self.n_users = dataset.n_users
        self.n_items = dataset.m_items
        self.embedding_dim = embedding_dim
        self.feat_embed_dim = feat_embed_dim
        self.ui_layers = ui_layers
        self.item_layers = item_layers
        self.topk = topk
        self.lambda_coeff = lambda_coeff
        self.device = device
        self.modality_names = list(modality_features.keys())

        self.user_embedding = nn.Embedding(self.n_users, embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.modality_embeddings = nn.ModuleDict()
        self.modality_transforms = nn.ModuleDict()
        self.original_item_graphs = {}

        for name, features in modality_features.items():
            feature_tensor = torch.as_tensor(features, dtype=torch.float32)
            if feature_tensor.ndim != 2 or feature_tensor.shape[0] != self.n_items:
                raise ValueError(
                    f"Feature matrix for {name} must have shape [n_items, dim]; "
                    f"got {tuple(feature_tensor.shape)}, expected first dim {self.n_items}."
                )
            self.modality_embeddings[name] = nn.Embedding.from_pretrained(
                feature_tensor,
                freeze=False,
            )
            self.modality_transforms[name] = nn.Linear(feature_tensor.shape[1], feat_embed_dim)

        self.modal_weight = nn.Parameter(torch.ones(len(self.modality_names), dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)

        self.user_item_graph = dataset.getSparseGraph().to(device)
        self.item_adj = None
        self._build_original_item_graphs()

    def _build_original_item_graphs(self):
        with torch.no_grad():
            for name in self.modality_names:
                features = self.modality_embeddings[name].weight.detach().to(self.device)
                sim = build_similarity(features)
                graph = build_knn_graph(sim, self.topk)
                self.original_item_graphs[name] = normalize_dense_adj(graph).detach()

    def _build_item_graph(self):
        weights = self.softmax(self.modal_weight)
        learned_adj = None
        original_adj = None

        for idx, name in enumerate(self.modality_names):
            features = self.modality_transforms[name](self.modality_embeddings[name].weight)
            sim = build_similarity(features)
            graph = build_knn_graph(sim, self.topk)
            norm_graph = normalize_dense_adj(graph)

            weighted_learned = weights[idx] * norm_graph
            weighted_original = weights[idx] * self.original_item_graphs[name].to(features.device)
            learned_adj = weighted_learned if learned_adj is None else learned_adj + weighted_learned
            original_adj = weighted_original if original_adj is None else original_adj + weighted_original

        return (1.0 - self.lambda_coeff) * learned_adj + self.lambda_coeff * original_adj

    def computer(self, build_item_graph=False):
        if build_item_graph or self.item_adj is None:
            self.item_adj = self._build_item_graph()
        else:
            self.item_adj = self.item_adj.detach()

        item_graph_emb = self.item_id_embedding.weight
        for _ in range(self.item_layers):
            item_graph_emb = torch.matmul(self.item_adj, item_graph_emb)

        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_id_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]
        graph = self.user_item_graph
        if graph.device != ego_embeddings.device:
            graph = graph.to(ego_embeddings.device)

        for _ in range(self.ui_layers):
            ego_embeddings = torch.sparse.mm(graph, ego_embeddings)
            all_embeddings.append(ego_embeddings)

        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        users, items = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        items = items + F.normalize(item_graph_emb, p=2, dim=1)
        return users, items

    def bpr_loss(self, users, pos_items, neg_items, build_item_graph=False):
        all_users, all_items = self.computer(build_item_graph=build_item_graph)
        users_emb = all_users[users.long()]
        pos_emb = all_items[pos_items.long()]
        neg_emb = all_items[neg_items.long()]

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        ranking_loss = F.softplus(neg_scores - pos_scores).mean()
        reg_loss = (
            users_emb.norm(2).pow(2)
            + pos_emb.norm(2).pow(2)
            + neg_emb.norm(2).pow(2)
        ) / (2.0 * users.size(0))
        return ranking_loss, reg_loss

    def forward(self, build_item_graph=False):
        return self.computer(build_item_graph=build_item_graph)

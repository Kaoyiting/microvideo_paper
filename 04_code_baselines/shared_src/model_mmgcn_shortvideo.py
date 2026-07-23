import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityGCN(nn.Module):
    def __init__(self, num_users, num_items, latent_dim, n_layers, features, device):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.device = device

        self.user_embedding = nn.Parameter(torch.empty(num_users, latent_dim))
        nn.init.xavier_uniform_(self.user_embedding)
        self.item_proj = nn.Linear(features.shape[1], latent_dim)
        self.register_buffer("item_features", torch.as_tensor(features, dtype=torch.float32))

    def forward(self, graph):
        user_feat = self.user_embedding
        item_feat = self.item_proj(self.item_features)
        all_feat = torch.cat([user_feat, item_feat], dim=0)

        layers = [all_feat]
        for _ in range(self.n_layers):
            all_feat = torch.sparse.mm(graph, all_feat)
            layers.append(all_feat)

        stacked = torch.stack(layers, dim=1)
        out = torch.mean(stacked, dim=1)
        users, items = torch.split(out, [self.num_users, self.num_items])
        return users, items


class MMGCN(nn.Module):
    """
    ShortVideo adaptation of MMGCN.
    Keeps the original three-modality idea:
    - visual
    - acoustic
    - textual

    Dataset adaptation only:
    - ShortVideo local embeddings are used as modality inputs.
    - The training objective remains pairwise BPR with 1:8 negative sampling
      in the companion baseline script.

    Reference:
    - MMGCN: Multi-modal Graph Convolution Network for Personalized Recommendation of Micro-video
    - Official implementation: https://github.com/weiyinwei/MMGCN

    This ShortVideo version keeps the three-modality structure and adapts the
    input pipeline to the local ShortVideo embeddings. The fusion is kept
    intentionally simple and deterministic so the baseline stays close to the
    paper-style multimodal aggregation behavior.
    """

    def __init__(self, config, dataset, modality_features):
        super().__init__()
        self.device = config["device"]
        self.n_users = dataset.n_users
        self.n_items = dataset.m_items
        self.latent_dim = config["latent_dim_rec"]
        self.n_layers = config["lightGCN_n_layers"]
        self.modality_names = list(modality_features.keys())
        self.n_modalities = len(self.modality_names)

        if self.n_modalities < 3:
            raise ValueError("MMGCN ShortVideo expects three modalities: visual, acoustic, textual.")

        self.graph = dataset.getSparseGraph().to(self.device)
        self.modality_gcns = nn.ModuleDict({
            name: ModalityGCN(
                self.n_users,
                self.n_items,
                self.latent_dim,
                self.n_layers,
                modality_features[name],
                self.device,
            ).to(self.device)
            for name in self.modality_names
        })

    def computer(self):
        user_reps = []
        item_reps = []
        for name in self.modality_names:
            users, items = self.modality_gcns[name](self.graph)
            user_reps.append(users)
            item_reps.append(items)

        user_final = torch.stack(user_reps, dim=0).mean(dim=0)
        item_final = torch.stack(item_reps, dim=0).mean(dim=0)
        return user_final, item_final

    def forward(self):
        return self.computer()

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        return torch.matmul(users_emb, all_items.t())

    def bpr_loss(self, users, pos, neg):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        pos_emb = all_items[pos.long()]
        neg_emb = all_items[neg.long()]

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(F.softplus(neg_scores - pos_scores))

        reg_loss = (
            users_emb.norm(2).pow(2)
            + pos_emb.norm(2).pow(2)
            + neg_emb.norm(2).pow(2)
        ) / (2.0 * users.size(0))
        return loss, reg_loss

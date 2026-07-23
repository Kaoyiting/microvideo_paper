import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, num_user, num_item, dim_id, layers, device, features):
        super(GCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_id = dim_id
        self.layers = layers
        self.device = device
        self.preference = nn.Parameter(torch.empty(num_user, dim_id))
        nn.init.xavier_normal_(self.preference)
        self.MLP = nn.Linear(features.shape[1], dim_id)
        self.register_buffer("features", features.float())

    def forward(self, edge_index):
        user_feat = self.preference
        item_feat = self.MLP(self.features)

        all_emb = torch.cat([user_feat, item_feat])
        embs = [all_emb]

        for _ in range(self.layers):
            all_emb = torch.sparse.mm(edge_index, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)

        user_gcn, item_gcn = torch.split(light_out, [self.num_user, self.num_item])
        return torch.cat([user_gcn, item_gcn]), user_gcn


class User_Graph_sample(nn.Module):
    def __init__(self, num_user, dim_latent):
        super(User_Graph_sample, self).__init__()
        self.num_user = num_user
        self.dim_latent = dim_latent

    def forward(self, features, user_graph, user_weight_matrix):
        u_features = features[user_graph]
        user_matrix = user_weight_matrix.unsqueeze(1)
        u_pre = torch.matmul(user_matrix, u_features)
        return u_pre.squeeze(1)


class DualGNN(nn.Module):
    def __init__(self, config, dataset, semantic_feat=None, content_feat=None, modality_features=None):
        super(DualGNN, self).__init__()
        self.device = config["device"]
        self.n_users = dataset.n_users
        self.n_items = dataset.m_items
        self.dim_latent = config["latent_dim_rec"]
        self.n_layers = config["lightGCN_n_layers"]

        if modality_features is None:
            if semantic_feat is None or content_feat is None:
                raise ValueError("Provide either modality_features or semantic/content features.")
            modality_features = {
                "semantic": semantic_feat,
                "content": content_feat,
            }

        if len(modality_features) < 2:
            raise ValueError("DualGNN requires at least two modalities.")

        self.modality_names = list(modality_features.keys())
        self.n_modalities = len(self.modality_names)
        self.modality_features = {
            name: feat.to(self.device).float()
            for name, feat in modality_features.items()
        }

        self.edge_index = dataset.getSparseGraph().to(self.device)

        # One user-item GCN branch per modality, matching DualGNN's
        # single-modal representation learning module.
        self.modality_gcns = nn.ModuleDict({
            name: GCN(
                self.n_users,
                self.n_items,
                self.dim_latent,
                self.n_layers,
                self.device,
                self.modality_features[name],
            ).to(self.device)
            for name in self.modality_names
        })

        # User/item-specific modality weights for adaptive multi-modal fusion.
        self.weight_u = nn.Parameter(torch.empty(self.n_users, self.n_modalities, 1))
        nn.init.xavier_normal_(self.weight_u)
        self.weight_i = nn.Parameter(torch.empty(self.n_items, self.n_modalities, 1))
        nn.init.xavier_normal_(self.weight_i)

        self.user_graph_model = User_Graph_sample(self.n_users, self.dim_latent).to(self.device)
        self.result_embed = None

    def computer(self, user_graph=None, user_weight_matrix=None):
        modality_reps = []
        modality_preferences = []

        for name in self.modality_names:
            rep, preference = self.modality_gcns[name](self.edge_index)
            modality_reps.append(rep)
            modality_preferences.append(preference)

        user_modal_reps = torch.stack(
            [rep[:self.n_users] for rep in modality_reps],
            dim=2,
        )
        item_modal_reps = torch.stack(
            [rep[self.n_users:] for rep in modality_reps],
            dim=2,
        )

        user_weights = F.softmax(self.weight_u, dim=1)
        item_weights = F.softmax(self.weight_i, dim=1)

        user_rep = torch.matmul(user_modal_reps, user_weights).squeeze(2)
        item_rep = torch.matmul(item_modal_reps, item_weights).squeeze(2)

        if user_graph is not None and user_weight_matrix is not None:
            h_u1 = self.user_graph_model(user_rep, user_graph, user_weight_matrix)
            user_rep = user_rep + h_u1

        self.result_embed = torch.cat((user_rep, item_rep), dim=0)
        return user_rep, item_rep, modality_preferences, modality_reps

    def getUsersRating(self, users, user_graph=None, user_weight_matrix=None):
        user_rep, item_rep, _, _ = self.computer(user_graph, user_weight_matrix)
        users_emb = user_rep[users.long()]
        return torch.matmul(users_emb, item_rep.t())

    def bpr_loss(self, users, pos, neg, user_graph=None, user_weight_matrix=None):
        user_rep, item_rep, modality_preferences, _ = self.computer(
            user_graph,
            user_weight_matrix,
        )

        users_emb = user_rep[users.long()]
        pos_emb = item_rep[pos.long()]
        neg_emb = item_rep[neg.long()]

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(F.softplus(neg_scores - pos_scores))

        reg_loss = torch.zeros((), device=self.device)
        for preference in modality_preferences:
            reg_loss = reg_loss + (preference[users.long()] ** 2).mean()

        reg_loss = reg_loss + (self.weight_u ** 2).mean()
        reg_loss = reg_loss + (self.weight_i ** 2).mean()

        return loss, reg_loss

"""
MMSSL model adapted from the official WWW 2023 implementation.

Reference:
    Wei, Huang, Xia, and Zhang (2023), "Multi-Modal Self-Supervised
    Learning for Recommendation", Proceedings of the ACM Web Conference 2023.

Official implementation:
    https://github.com/HKUDS/MMSSL

Fidelity notes:
    - The neural architecture, forward data flow, multi-head cross-modal
      attention, BPR objective, feature regularization, adversarial
      discriminator loss, and cross-modal contrastive loss follow the official
      MMSSL/Models.py and MMSSL/main.py training path.
    - The official implementation is hard-coded for CUDA and imports DGL. This
      local version removes hard-coded CUDA calls and does not use DGL because
      the current project environment is Python 3.13 CPU-only and has no DGL
      package. The DGL helper functions in the official code are not used by
      the main training loss path.
    - The official code has image/text encoders. To stay source-faithful, this
      model consumes two modalities named image and text; dataset scripts map
      local visual features to image and textual features to text. Acoustic
      features are therefore not used in this source-faithful baseline.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd


@dataclass
class MMSSLConfig:
    embed_size: int = 64
    weight_size: tuple = (64, 64)
    layers: int = 1
    drop_rate: float = 0.2
    head_num: int = 4
    model_cat_rate: float = 0.55
    id_cat_rate: float = 0.36
    feat_reg_decay: float = 1e-5
    cl_rate: float = 0.03
    g_rate: float = 0.0001
    g_drop1: float = 0.31
    g_drop2: float = 0.5
    gp_rate: float = 1.0
    real_data_tau: float = 0.005
    log_log_scale: float = 0.00001
    ui_pre_scale: float = 100.0
    m_topk_rate: float = 0.0001
    t_update: int = 1
    tau: float = 0.5
    decay: float = 1e-5
    batch_size: int = 1024


def csr_norm(csr_mat, mean_flag=False):
    rowsum = np.array(csr_mat.sum(1))
    rowsum = np.power(rowsum + 1e-8, -0.5).flatten()
    rowsum[np.isinf(rowsum)] = 0.0
    rowsum_diag = sp.diags(rowsum)

    colsum = np.array(csr_mat.sum(0))
    colsum = np.power(colsum + 1e-8, -0.5).flatten()
    colsum[np.isinf(colsum)] = 0.0
    colsum_diag = sp.diags(colsum)

    if mean_flag:
        return rowsum_diag * csr_mat
    return rowsum_diag * csr_mat * colsum_diag


def sparse_mx_to_torch_sparse_tensor(sparse_mx, device):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32, device=device).coalesce()


class MMSSL(nn.Module):
    def __init__(self, n_users, n_items, config, image_feats, text_feats, device):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.config = config
        self.device = device
        self.embedding_dim = config.embed_size
        self.weight_size = [config.embed_size] + list(config.weight_size)
        self.n_ui_layers = len(config.weight_size)

        self.image_trans = nn.Linear(image_feats.shape[1], config.embed_size)
        self.text_trans = nn.Linear(text_feats.shape[1], config.embed_size)
        nn.init.xavier_uniform_(self.image_trans.weight)
        nn.init.xavier_uniform_(self.text_trans.weight)

        self.common_trans = nn.Linear(config.embed_size, config.embed_size)
        nn.init.xavier_uniform_(self.common_trans.weight)

        self.user_id_embedding = nn.Embedding(n_users, config.embed_size)
        self.item_id_embedding = nn.Embedding(n_items, config.embed_size)
        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.image_embedding = nn.Embedding.from_pretrained(
            torch.as_tensor(image_feats, dtype=torch.float32),
            freeze=False,
        )
        self.text_embedding = nn.Embedding.from_pretrained(
            torch.as_tensor(text_feats, dtype=torch.float32),
            freeze=False,
        )

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=config.drop_rate)
        self.tau = config.tau

        initializer = nn.init.xavier_uniform_
        self.weight_dict = nn.ParameterDict({
            "w_q": nn.Parameter(initializer(torch.empty([config.embed_size, config.embed_size]))),
            "w_k": nn.Parameter(initializer(torch.empty([config.embed_size, config.embed_size]))),
            "w_v": nn.Parameter(initializer(torch.empty([config.embed_size, config.embed_size]))),
            "w_self_attention_item": nn.Parameter(initializer(torch.empty([config.embed_size, config.embed_size]))),
            "w_self_attention_user": nn.Parameter(initializer(torch.empty([config.embed_size, config.embed_size]))),
            "w_self_attention_cat": nn.Parameter(initializer(torch.empty([config.head_num * config.embed_size, config.embed_size]))),
        })
        self.embedding_dict = {"user": {}, "item": {}}

    def mm(self, x, y):
        if x.is_sparse:
            return torch.sparse.mm(x, y)
        return torch.mm(x, y)

    def para_dict_to_tensor(self, para_dict):
        return torch.stack([para_dict[key] for key in para_dict.keys()], dim=0)

    def multi_head_self_attention(self, trans_w, embedding_t_1, embedding_t):
        q = self.para_dict_to_tensor(embedding_t)
        v = k = self.para_dict_to_tensor(embedding_t_1)
        beh, n_nodes = q.shape[0], q.shape[1]
        d_h = self.config.embed_size / self.config.head_num

        q_proj = torch.matmul(q, trans_w["w_q"])
        _ = torch.matmul(k, trans_w["w_k"])
        _ = torch.matmul(v, trans_w["w_v"])

        q_heads = q_proj.reshape(beh, n_nodes, self.config.head_num, int(d_h)).permute(2, 0, 1, 3)
        # The official code assigns K from Q after projection. Keep this behavior
        # for source fidelity instead of silently correcting it.
        k_heads = q_proj.reshape(beh, n_nodes, self.config.head_num, int(d_h)).permute(2, 0, 1, 3)
        v_heads = v

        q_heads = torch.unsqueeze(q_heads, 2)
        k_heads = torch.unsqueeze(k_heads, 1)
        v_heads = torch.unsqueeze(v_heads, 1)

        att = torch.mul(q_heads, k_heads) / torch.sqrt(torch.tensor(d_h, device=q.device))
        att = torch.sum(att, dim=-1)
        att = torch.unsqueeze(att, dim=-1)
        att = F.softmax(att, dim=2)

        z = torch.mul(att, v_heads)
        z = torch.sum(z, dim=2)
        z = torch.cat([value for value in z], dim=-1)
        z = torch.matmul(z, self.weight_dict["w_self_attention_cat"])
        return z, att.detach()

    def forward(self, ui_graph, iu_graph, image_ui_graph, image_iu_graph, text_ui_graph, text_iu_graph):
        image_feats = image_item_feats = self.dropout(self.image_trans(self.image_embedding.weight))
        text_feats = text_item_feats = self.dropout(self.text_trans(self.text_embedding.weight))

        for _ in range(self.config.layers):
            image_user_feats = self.mm(ui_graph, image_feats)
            image_item_feats = self.mm(iu_graph, image_user_feats)
            image_user_id = self.mm(image_ui_graph, self.item_id_embedding.weight)
            image_item_id = self.mm(image_iu_graph, self.user_id_embedding.weight)

            text_user_feats = self.mm(ui_graph, text_feats)
            text_item_feats = self.mm(iu_graph, text_user_feats)
            text_user_id = self.mm(text_ui_graph, self.item_id_embedding.weight)
            text_item_id = self.mm(text_iu_graph, self.user_id_embedding.weight)

        self.embedding_dict["user"]["image"] = image_user_id
        self.embedding_dict["user"]["text"] = text_user_id
        self.embedding_dict["item"]["image"] = image_item_id
        self.embedding_dict["item"]["text"] = text_item_id

        user_z, _ = self.multi_head_self_attention(self.weight_dict, self.embedding_dict["user"], self.embedding_dict["user"])
        item_z, _ = self.multi_head_self_attention(self.weight_dict, self.embedding_dict["item"], self.embedding_dict["item"])
        user_emb = user_z.mean(0)
        item_emb = item_z.mean(0)

        u_g_embeddings = self.user_id_embedding.weight + self.config.id_cat_rate * F.normalize(user_emb, p=2, dim=1)
        i_g_embeddings = self.item_id_embedding.weight + self.config.id_cat_rate * F.normalize(item_emb, p=2, dim=1)

        user_emb_list = [u_g_embeddings]
        item_emb_list = [i_g_embeddings]
        for layer_idx in range(self.n_ui_layers):
            if layer_idx == self.n_ui_layers - 1:
                u_g_embeddings = self.softmax(self.mm(ui_graph, i_g_embeddings))
                i_g_embeddings = self.softmax(self.mm(iu_graph, u_g_embeddings))
            else:
                u_g_embeddings = self.mm(ui_graph, i_g_embeddings)
                i_g_embeddings = self.mm(iu_graph, u_g_embeddings)
            user_emb_list.append(u_g_embeddings)
            item_emb_list.append(i_g_embeddings)

        u_g_embeddings = torch.mean(torch.stack(user_emb_list), dim=0)
        i_g_embeddings = torch.mean(torch.stack(item_emb_list), dim=0)

        u_g_embeddings = (
            u_g_embeddings
            + self.config.model_cat_rate * F.normalize(image_user_feats, p=2, dim=1)
            + self.config.model_cat_rate * F.normalize(text_user_feats, p=2, dim=1)
        )
        i_g_embeddings = (
            i_g_embeddings
            + self.config.model_cat_rate * F.normalize(image_item_feats, p=2, dim=1)
            + self.config.model_cat_rate * F.normalize(text_item_feats, p=2, dim=1)
        )

        return (
            u_g_embeddings,
            i_g_embeddings,
            image_item_feats,
            text_item_feats,
            image_user_feats,
            text_user_feats,
            u_g_embeddings,
            i_g_embeddings,
            image_user_id,
            text_user_id,
            image_item_id,
            text_item_id,
        )


class Discriminator(nn.Module):
    def __init__(self, dim, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, int(dim / 4)),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(int(dim / 4)),
            nn.Dropout(config.g_drop1),
            nn.Linear(int(dim / 4), int(dim / 8)),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(int(dim / 8)),
            nn.Dropout(config.g_drop2),
            nn.Linear(int(dim / 8), 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return (100 * self.net(x.float())).view(-1)


def bpr_loss(users, pos_items, neg_items, batch_size, decay):
    pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
    neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
    regularizer = 0.5 * (users ** 2).sum() + 0.5 * (pos_items ** 2).sum() + 0.5 * (neg_items ** 2).sum()
    regularizer = regularizer / batch_size
    mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
    emb_loss = decay * regularizer
    reg_loss = 0.0
    return mf_loss, emb_loss, reg_loss


def sim(z1, z2):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


def batched_contrastive_loss(z1, z2, tau=0.5, batch_size=1024):
    device = z1.device
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1
    f = lambda x: torch.exp(x / tau)
    indices = torch.arange(0, num_nodes, device=device)
    losses = []

    for batch_idx in range(num_batches):
        tmp_i = indices[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        refl_sim_list = []
        between_sim_list = []
        for other_idx in range(num_batches):
            tmp_j = indices[other_idx * batch_size:(other_idx + 1) * batch_size]
            refl_sim_list.append(f(sim(z1[tmp_i], z1[tmp_j])))
            between_sim_list.append(f(sim(z1[tmp_i], z2[tmp_j])))

        refl_sim = torch.cat(refl_sim_list, dim=-1)
        between_sim = torch.cat(between_sim_list, dim=-1)
        losses.append(
            -torch.log(
                between_sim[:, batch_idx * batch_size:(batch_idx + 1) * batch_size].diag()
                / (
                    refl_sim.sum(1)
                    + between_sim.sum(1)
                    - refl_sim[:, batch_idx * batch_size:(batch_idx + 1) * batch_size].diag()
                )
                + 1e-8
            )
        )

    return torch.cat(losses).mean()


def feat_reg_loss(g_item_image, g_item_text, g_user_image, g_user_text, n_items, feat_reg_decay):
    feat_reg = (
        0.5 * (g_item_image ** 2).sum()
        + 0.5 * (g_item_text ** 2).sum()
        + 0.5 * (g_user_image ** 2).sum()
        + 0.5 * (g_user_text ** 2).sum()
    )
    feat_reg = feat_reg / n_items
    return feat_reg_decay * feat_reg


def gradient_penalty(discriminator, xr, xf, batch_size, device):
    penalty_lambda = 0.3
    xf = xf.detach()
    xr = xr.detach()
    alpha = torch.rand(batch_size * 2, 1, device=device)
    alpha = alpha.expand_as(xr)
    interpolates = alpha * xr + ((1 - alpha) * xf)
    interpolates.requires_grad_()
    disc_interpolates = discriminator(interpolates)
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * penalty_lambda

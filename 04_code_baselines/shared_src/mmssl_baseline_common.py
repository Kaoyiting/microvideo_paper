#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared MMSSL baseline runner utilities.

This module follows the official MMSSL training path while adapting data IO and
evaluation to this thesis repository. Dataset-specific scripts must clearly
document their feature mapping and candidate item space.
"""

import json
import os
import random
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import torch
import torch.nn.functional as F

from model_mmssl import (
    Discriminator,
    MMSSL,
    MMSSLConfig,
    batched_contrastive_loss,
    bpr_loss,
    csr_norm,
    feat_reg_loss,
    gradient_penalty,
    sparse_mx_to_torch_sparse_tensor,
)
from unified_evaluator import print_results


K_LIST = [5, 10, 20]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_user_item_dict(users, items):
    user_item_dict = {}
    for user, item in zip(users, items):
        user_item_dict.setdefault(int(user), []).append(int(item))
    return user_item_dict


def normalize_rows(matrix):
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)


def load_visual_textual_features(path, num_items):
    data = np.load(path, allow_pickle=True)
    required = ["valid_ids", "visual", "textual"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}")

    valid_ids = np.asarray(data["valid_ids"], dtype=np.int64)
    if len(valid_ids) != num_items or not np.array_equal(valid_ids, np.arange(num_items)):
        raise ValueError("MMSSL expects contiguous valid_ids covering all dataset items.")

    image_feats = normalize_rows(np.asarray(data["visual"], dtype=np.float32))
    text_feats = normalize_rows(np.asarray(data["textual"], dtype=np.float32))
    return image_feats, text_feats


def load_id_mappings(id_mapping_path):
    with open(id_mapping_path, "r", encoding="utf-8") as f:
        id_mappings = json.load(f)
    item_id_to_dataset_idx = {
        str(raw_item_id): int(dataset_idx)
        for raw_item_id, dataset_idx in id_mappings["item_to_idx"].items()
    }
    dataset_idx_to_item_id = {
        int(dataset_idx): str(raw_item_id)
        for raw_item_id, dataset_idx in id_mappings["item_to_idx"].items()
    }
    return item_id_to_dataset_idx, dataset_idx_to_item_id


def load_shortvideo_aligned(aligned_embeddings_path, num_items):
    data = np.load(aligned_embeddings_path, allow_pickle=True)
    if "valid_ids" not in data.files:
        raise KeyError(f"{aligned_embeddings_path} must contain valid_ids.")
    valid_ids = np.asarray(data["valid_ids"], dtype=np.int64)
    valid_ids = valid_ids[(valid_ids >= 0) & (valid_ids < num_items)]
    raw_id_to_aligned_idx = {int(item_id): idx for idx, item_id in enumerate(valid_ids.tolist())}
    return valid_ids, raw_id_to_aligned_idx, None, 0, len(valid_ids)


def load_microlens_aligned(aligned_embeddings_path, id_mapping_path):
    data = np.load(aligned_embeddings_path, allow_pickle=True)
    if "item_ids" not in data.files:
        raise KeyError(f"{aligned_embeddings_path} must contain item_ids.")
    item_id_to_dataset_idx, dataset_idx_to_item_id = load_id_mappings(id_mapping_path)
    raw_item_ids = [str(item_id) for item_id in data["item_ids"]]
    aligned_dataset_indices = []
    raw_id_to_aligned_idx = {}
    missing_count = 0
    for raw_item_id in raw_item_ids:
        dataset_idx = item_id_to_dataset_idx.get(raw_item_id)
        if dataset_idx is None:
            missing_count += 1
            continue
        raw_id_to_aligned_idx[raw_item_id] = len(aligned_dataset_indices)
        aligned_dataset_indices.append(dataset_idx)
    return np.asarray(aligned_dataset_indices, dtype=np.int64), raw_id_to_aligned_idx, dataset_idx_to_item_id, missing_count, len(raw_item_ids)


class MMSSLTrainer:
    def __init__(self, dataset, image_feats, text_feats, args, device):
        self.dataset = dataset
        self.args = args
        self.device = device
        self.config = MMSSLConfig(
            embed_size=args.embed_size,
            weight_size=tuple(args.weight_size),
            layers=args.layers,
            drop_rate=args.drop_rate,
            head_num=args.head_num,
            model_cat_rate=args.model_cat_rate,
            id_cat_rate=args.id_cat_rate,
            feat_reg_decay=args.feat_reg_decay,
            cl_rate=args.cl_rate,
            g_rate=args.g_rate,
            g_drop1=args.g_drop1,
            g_drop2=args.g_drop2,
            gp_rate=args.gp_rate,
            real_data_tau=args.real_data_tau,
            log_log_scale=args.log_log_scale,
            ui_pre_scale=args.ui_pre_scale,
            m_topk_rate=args.m_topk_rate,
            t_update=args.t_update,
            tau=args.tau,
            decay=args.decay,
            batch_size=args.batch_size,
        )

        self.ui_graph_raw = dataset.UserItemNet.tocsr().astype(np.float32)
        self.iu_graph_raw = self.ui_graph_raw.T.tocsr()
        self.ui_graph = sparse_mx_to_torch_sparse_tensor(csr_norm(self.ui_graph_raw, mean_flag=True), device)
        self.iu_graph = sparse_mx_to_torch_sparse_tensor(csr_norm(self.iu_graph_raw, mean_flag=True), device)
        self.image_ui_graph = self.ui_graph
        self.image_iu_graph = self.iu_graph
        self.text_ui_graph = self.ui_graph
        self.text_iu_graph = self.iu_graph
        self.image_ui_index = {"x": [], "y": []}
        self.text_ui_index = {"x": [], "y": []}

        self.exist_users = np.asarray([u for u, items in enumerate(dataset.allPos) if len(items) > 0], dtype=np.int64)
        self.model = MMSSL(dataset.n_users, dataset.m_items, self.config, image_feats, text_feats, device).to(device)
        self.discriminator = Discriminator(dataset.m_items, self.config).to(device)
        self.discriminator.apply(self.weights_init)
        self.optim_d = torch.optim.Adam(self.discriminator.parameters(), lr=args.d_lr, betas=(0.5, 0.9))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)

    @staticmethod
    def weights_init(module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0)

    def sample_batch(self):
        if self.args.batch_size <= len(self.exist_users):
            users = np.random.choice(self.exist_users, size=self.args.batch_size, replace=False)
        else:
            users = np.random.choice(self.exist_users, size=self.args.batch_size, replace=True)
        pos_items, neg_items = [], []
        for user in users:
            positives = self.dataset.allPos[int(user)]
            pos_items.append(int(positives[np.random.randint(0, len(positives))]))
            while True:
                neg_item = np.random.randint(0, self.dataset.m_items)
                if neg_item not in positives:
                    neg_items.append(int(neg_item))
                    break
        return users.astype(np.int64), np.asarray(pos_items, dtype=np.int64), np.asarray(neg_items, dtype=np.int64)

    def u_sim_calculation(self, users, user_final, item_final):
        topk_u = user_final[users]
        u_ui = torch.as_tensor(self.ui_graph_raw[users].toarray(), dtype=torch.float32, device=self.device)
        num_batches = (self.dataset.m_items - 1) // self.args.batch_size + 1
        indices = torch.arange(0, self.dataset.m_items, device=self.device)
        u_sim_list = []
        for batch_idx in range(num_batches):
            index = indices[batch_idx * self.args.batch_size:(batch_idx + 1) * self.args.batch_size]
            sim = torch.mm(topk_u, item_final[index].t())
            sim_gt = torch.multiply(sim, (1 - u_ui[:, index]))
            u_sim_list.append(sim_gt)
        return F.normalize(torch.cat(u_sim_list, dim=-1), p=2, dim=1)

    def update_generated_graphs_if_needed(self, batch_idx):
        if batch_idx % self.args.t_update != 0 or batch_idx == 0:
            return
        self.image_ui_graph = self._index_to_graph(self.image_ui_index)
        self.text_ui_graph = self._index_to_graph(self.text_ui_index)
        self.image_iu_graph = self._transpose_ui_graph(self.image_ui_index)
        self.text_iu_graph = self._transpose_ui_graph(self.text_ui_index)
        self.image_ui_index = {"x": [], "y": []}
        self.text_ui_index = {"x": [], "y": []}

    def _index_to_graph(self, index_dict):
        if not index_dict["x"]:
            return self.ui_graph
        graph = csr_matrix(
            (np.ones(len(index_dict["x"]), dtype=np.float32), (index_dict["x"], index_dict["y"])),
            shape=(self.dataset.n_users, self.dataset.m_items),
        )
        return sparse_mx_to_torch_sparse_tensor(csr_norm(graph, mean_flag=True), self.device)

    def _transpose_ui_graph(self, index_dict):
        if not index_dict["x"]:
            return self.iu_graph
        graph = csr_matrix(
            (np.ones(len(index_dict["x"]), dtype=np.float32), (index_dict["x"], index_dict["y"])),
            shape=(self.dataset.n_users, self.dataset.m_items),
        ).T
        return sparse_mx_to_torch_sparse_tensor(csr_norm(graph, mean_flag=True), self.device)

    def collect_generated_edges(self, users, image_u_sim, text_u_sim):
        topk_count = max(1, int(self.dataset.m_items * self.args.m_topk_rate))
        _, image_ids = torch.topk(image_u_sim, topk_count, dim=-1)
        _, text_ids = torch.topk(text_u_sim, topk_count, dim=-1)
        user_repeated = torch.as_tensor(users, device=self.device).unsqueeze(1).repeat(1, topk_count)
        self.image_ui_index["x"] += user_repeated.reshape(-1).cpu().numpy().astype(int).tolist()
        self.image_ui_index["y"] += image_ids.reshape(-1).cpu().numpy().astype(int).tolist()
        self.text_ui_index["x"] += user_repeated.reshape(-1).cpu().numpy().astype(int).tolist()
        self.text_ui_index["y"] += text_ids.reshape(-1).cpu().numpy().astype(int).tolist()

    def train_one_epoch(self):
        self.model.train()
        n_batch = self.dataset.trainDataSize // self.args.batch_size + 1
        max_batches = getattr(self.args, "max_batches", 0)
        if max_batches and max_batches > 0:
            n_batch = min(n_batch, max_batches)
        total_loss = 0.0

        for batch_idx in range(n_batch):
            users_np, pos_np, neg_np = self.sample_batch()
            users = torch.LongTensor(users_np).to(self.device)
            pos_items = torch.LongTensor(pos_np).to(self.device)
            neg_items = torch.LongTensor(neg_np).to(self.device)

            with torch.no_grad():
                ua, ia, image_item, text_item, image_user, text_user, *_ = self.model(
                    self.ui_graph, self.iu_graph,
                    self.image_ui_graph, self.image_iu_graph,
                    self.text_ui_graph, self.text_iu_graph,
                )
            ui_u_sim = self.u_sim_calculation(users_np, ua, ia).detach()
            image_u_sim = self.u_sim_calculation(users_np, image_user, image_item).detach()
            text_u_sim = self.u_sim_calculation(users_np, text_user, text_item).detach()

            input_fake = torch.cat((image_u_sim, text_u_sim), dim=0)
            pred_fake = self.discriminator(input_fake)
            loss_fake = pred_fake.mean()

            u_ui = torch.as_tensor(self.ui_graph_raw[users_np].toarray(), dtype=torch.float32, device=self.device)
            gumbel_noise = -torch.log(torch.empty_like(u_ui).uniform_(0, 1) + 1e-8)
            u_ui = F.softmax(
                u_ui - self.args.log_log_scale * torch.log(gumbel_noise + 1e-8) / self.args.real_data_tau,
                dim=1,
            )
            u_ui = F.normalize(u_ui + ui_u_sim * self.args.ui_pre_scale, dim=1)
            input_real = torch.cat((u_ui, u_ui), dim=0)
            pred_real = self.discriminator(input_real)
            loss_real = -pred_real.mean()
            gp = gradient_penalty(self.discriminator, input_real, input_fake.detach(), len(users_np), self.device)
            loss_d = loss_real + loss_fake + self.args.gp_rate * gp
            self.optim_d.zero_grad()
            loss_d.backward()
            self.optim_d.step()

            outputs = self.model(
                self.ui_graph, self.iu_graph,
                self.image_ui_graph, self.image_iu_graph,
                self.text_ui_graph, self.text_iu_graph,
            )
            (
                g_ua, g_ia, g_image_item, g_text_item,
                g_image_user, g_text_user, g_user_emb, _,
                g_image_user_id, g_text_user_id, _, _,
            ) = outputs
            g_users = g_ua[users]
            g_pos = g_ia[pos_items]
            g_neg = g_ia[neg_items]
            mf_loss, emb_loss, reg_loss = bpr_loss(g_users, g_pos, g_neg, self.args.batch_size, self.args.decay)

            g_image_u_sim = self.u_sim_calculation(users_np, g_image_user, g_image_item)
            g_text_u_sim = self.u_sim_calculation(users_np, g_text_user, g_text_item)
            self.update_generated_graphs_if_needed(batch_idx)
            if batch_idx % self.args.t_update != 0 or batch_idx == 0:
                self.collect_generated_edges(users_np, g_image_u_sim.detach(), g_text_u_sim.detach())

            feat_loss = feat_reg_loss(
                g_image_item, g_text_item, g_image_user, g_text_user,
                self.dataset.m_items, self.args.feat_reg_decay,
            )
            contrastive_loss = (
                batched_contrastive_loss(g_image_user_id[users], g_user_emb[users], tau=self.args.tau, batch_size=self.args.cl_batch_size)
                + batched_contrastive_loss(g_text_user_id[users], g_user_emb[users], tau=self.args.tau, batch_size=self.args.cl_batch_size)
            )
            g_input_fake = torch.cat((g_image_u_sim, g_text_u_sim), dim=0)
            g_loss_fake = -self.discriminator(g_input_fake).mean()
            batch_loss = (
                mf_loss + emb_loss + reg_loss + feat_loss
                + self.args.cl_rate * contrastive_loss
                + self.args.g_rate * g_loss_fake
            )
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
            total_loss += float(batch_loss.detach().cpu())

        return total_loss / max(n_batch, 1)

    def embeddings(self):
        self.model.eval()
        with torch.no_grad():
            ua, ia, *_ = self.model(
                self.ui_graph, self.iu_graph,
                self.image_ui_graph, self.image_iu_graph,
                self.text_ui_graph, self.text_iu_graph,
            )
        return ua, ia


def build_eval_data_shortvideo(dataset, item_id_to_aligned_idx):
    test_items_dict = build_user_item_dict(dataset.testUser, dataset.testItem)
    train_items_dict = build_user_item_dict(dataset.trainUser, dataset.trainItem)
    eval_users, user_test_items, user_train_items = [], [], []
    for user in sorted(test_items_dict):
        test_filtered = [item_id_to_aligned_idx[item] for item in test_items_dict[user] if item in item_id_to_aligned_idx]
        if not test_filtered:
            continue
        train_filtered = [item_id_to_aligned_idx[item] for item in train_items_dict.get(user, []) if item in item_id_to_aligned_idx]
        eval_users.append(user)
        user_test_items.append(test_filtered)
        user_train_items.append(train_filtered)
    return eval_users, user_test_items, user_train_items


def build_eval_data_microlens(dataset, raw_id_to_aligned_idx, dataset_idx_to_item_id):
    test_items_dict = build_user_item_dict(dataset.testUser, dataset.testItem)
    train_items_dict = build_user_item_dict(dataset.trainUser, dataset.trainItem)
    eval_users, user_test_items, user_train_items = [], [], []
    for user in sorted(test_items_dict):
        test_filtered = []
        for item_idx in test_items_dict[user]:
            raw_item_id = dataset_idx_to_item_id.get(item_idx)
            if raw_item_id in raw_id_to_aligned_idx:
                test_filtered.append(raw_id_to_aligned_idx[raw_item_id])
        if not test_filtered:
            continue
        train_filtered = []
        for item_idx in train_items_dict.get(user, []):
            raw_item_id = dataset_idx_to_item_id.get(item_idx)
            if raw_item_id in raw_id_to_aligned_idx:
                train_filtered.append(raw_id_to_aligned_idx[raw_item_id])
        eval_users.append(user)
        user_test_items.append(test_filtered)
        user_train_items.append(train_filtered)
    return eval_users, user_test_items, user_train_items


def compute_metrics_from_topk(top_indices, user_test_items):
    results = {f"{metric}@{k}": 0.0 for metric in ["Hit", "NDCG", "MRR"] for k in K_LIST}
    valid_users = 0
    for user_idx, ground_truth in enumerate(user_test_items):
        if not ground_truth:
            continue
        valid_users += 1
        gt_set = set(ground_truth)
        pred_items = top_indices[user_idx]
        for k in K_LIST:
            current_top_k = pred_items[:k]
            first_hit_rank = -1
            for rank, item in enumerate(current_top_k):
                if item in gt_set and first_hit_rank == -1:
                    first_hit_rank = rank + 1
            results[f"Hit@{k}"] += 1.0 if first_hit_rank != -1 else 0.0
            if first_hit_rank != -1:
                results[f"MRR@{k}"] += 1.0 / first_hit_rank
            dcg = sum(1.0 / np.log2(rank + 2) for rank, item in enumerate(current_top_k) if item in gt_set)
            idcg = sum(1.0 / np.log2(rank + 2) for rank in range(min(k, len(gt_set))))
            if idcg > 0:
                results[f"NDCG@{k}"] += dcg / idcg
    for key in results:
        results[key] /= max(valid_users, 1)
    return results


def evaluate_embeddings(user_embs, item_embs, eval_users, candidate_items, user_test_items, user_train_items, device):
    with torch.no_grad():
        user_indices = torch.LongTensor(eval_users).to(device)
        item_indices = torch.LongTensor(candidate_items).to(device)
        scores = torch.matmul(user_embs[user_indices], item_embs[item_indices].t())
        for row_idx, train_items in enumerate(user_train_items):
            if train_items:
                scores[row_idx, train_items] = -1e9
        _, top_indices = torch.topk(scores, max(K_LIST), dim=1)
    return compute_metrics_from_topk(top_indices.cpu().numpy(), user_test_items)


def run_training(args, dataset, image_feats, text_feats, aligned_items, eval_data, config_extra):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    trainer = MMSSLTrainer(dataset, image_feats, text_feats, args, device)

    config = vars(args).copy()
    config.update(config_extra)
    config.update({
        "official_paper": "Wei et al. (2023), Multi-Modal Self-Supervised Learning for Recommendation, WWW 2023",
        "official_implementation": "https://github.com/HKUDS/MMSSL",
        "source_fidelity": "Follows official Models.py/main.py training path; DGL helper not used because current environment lacks DGL.",
        "modalities": "official image/text encoders mapped to local visual/textual features",
        "mask_training_items": True,
    })

    best_hit20 = -1.0
    best_metrics = None
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        avg_loss = trainer.train_one_epoch()
        print(f"Epoch [{epoch:03d}/{args.epochs}] loss={avg_loss:.4f} time={time.time() - start:.2f}s")
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            user_embs, item_embs = trainer.embeddings()
            eval_users, user_test_items, user_train_items = eval_data
            metrics = evaluate_embeddings(
                user_embs,
                item_embs,
                eval_users,
                aligned_items,
                user_test_items,
                user_train_items,
                device=device,
            )
            print(
                f"Hit@20={metrics.get('Hit@20', 0.0):.4f}, "
                f"NDCG@20={metrics.get('NDCG@20', 0.0):.4f}, "
                f"MRR@20={metrics.get('MRR@20', 0.0):.4f}"
            )
            if metrics.get("Hit@20", 0.0) > best_hit20:
                best_hit20 = metrics["Hit@20"]
                best_metrics = metrics
                torch.save({"model_state_dict": trainer.model.state_dict(), "config": config, "metrics": best_metrics}, args.output)
                print("Saved best MMSSL checkpoint.")

    print_results(best_metrics or {}, config_extra.get("model_name", "MMSSL"))

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared MetaMMF-GCN baseline utilities."""

import json
import os
import random
import time

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, TensorDataset

from model_metammf import MetaMMFConfig, MetaMMFGCN, normalize_rows
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


def load_three_modal_features(path, num_items):
    data = np.load(path, allow_pickle=True)
    required = ["valid_ids", "visual", "acoustic", "textual"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}")
    valid_ids = np.asarray(data["valid_ids"], dtype=np.int64)
    if len(valid_ids) != num_items or not np.array_equal(valid_ids, np.arange(num_items)):
        raise ValueError("MetaMMF-GCN expects contiguous valid_ids covering all dataset items.")
    return {
        "visual": np.asarray(data["visual"], dtype=np.float32),
        "acoustic": np.asarray(data["acoustic"], dtype=np.float32),
        "textual": np.asarray(data["textual"], dtype=np.float32),
    }


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


def build_user_item_dict(users, items):
    user_item_dict = {}
    for user, item in zip(users, items):
        user_item_dict.setdefault(int(user), []).append(int(item))
    return user_item_dict


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


def build_mean_sage_adj(dataset, device):
    users = np.asarray(dataset.trainUser, dtype=np.int64)
    items = np.asarray(dataset.trainItem, dtype=np.int64) + dataset.n_users
    rows = np.concatenate([items, users])
    cols = np.concatenate([users, items])
    deg = np.bincount(rows, minlength=dataset.n_users + dataset.m_items).astype(np.float32)
    data = 1.0 / np.maximum(deg[rows], 1.0)
    adj = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(dataset.n_users + dataset.m_items, dataset.n_users + dataset.m_items),
        dtype=np.float32,
    )
    indices = torch.from_numpy(np.vstack([adj.row, adj.col]).astype(np.int64))
    values = torch.from_numpy(adj.data)
    return torch.sparse_coo_tensor(indices, values, adj.shape, dtype=torch.float32, device=device).coalesce()


def sample_bpr_triplets(dataset, negative_ratio):
    triplets = []
    all_pos = dataset.allPos
    item_ids = np.arange(dataset.m_items)
    candidate_cache = {}
    for user, pos_item in zip(dataset.trainUser, dataset.trainItem):
        user = int(user)
        pos_item = int(pos_item)
        user_pos = all_pos[user]
        if user not in candidate_cache:
            candidate_cache[user] = item_ids[~np.isin(item_ids, user_pos)]
        can_item_ids = candidate_cache[user]
        replace = len(can_item_ids) < negative_ratio
        neg_items = np.random.choice(can_item_ids, size=negative_ratio, replace=replace)
        for neg_item in neg_items:
            triplets.append((user, pos_item, int(neg_item)))
    return np.asarray(triplets, dtype=np.int64)


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


def evaluate(model, mean_adj, aligned_items, eval_data, device, fusion_batch_size):
    model.eval()
    eval_users, user_test_items, user_train_items = eval_data
    with torch.no_grad():
        all_users, all_items = model(mean_adj, fusion_batch_size=fusion_batch_size)
        user_indices = torch.LongTensor(eval_users).to(device)
        item_indices = torch.LongTensor(aligned_items).to(device)
        scores = torch.matmul(all_users[user_indices], all_items[item_indices].t())
        for row_idx, train_items in enumerate(user_train_items):
            if train_items:
                scores[row_idx, train_items] = -1e9
        _, top_indices = torch.topk(scores, max(K_LIST), dim=1)
    return compute_metrics_from_topk(top_indices.cpu().numpy(), user_test_items), len(eval_users)


def train_one_epoch(model, dataset, mean_adj, optimizer, args, device):
    model.train()
    samples = sample_bpr_triplets(dataset, args.negative_ratio)
    tensors = [torch.LongTensor(samples[:, idx]) for idx in range(3)]
    loader = DataLoader(
        TensorDataset(*tensors),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    total_loss = 0.0
    total_batches = 0
    for batch_idx, (users, pos_items, neg_items) in enumerate(loader):
        max_batches = getattr(args, "max_batches", 0)
        if max_batches and batch_idx >= max_batches:
            break
        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)
        loss = model.bpr_loss(
            users,
            pos_items,
            neg_items,
            mean_adj,
            fusion_batch_size=args.fusion_batch_size,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        total_batches += 1
    return total_loss / max(total_batches, 1)


def run_training(args, dataset, features, aligned_items, eval_data, config_extra):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    mean_adj = build_mean_sage_adj(dataset, device)
    config_obj = MetaMMFConfig(
        rank=args.rank,
        signal_dim=args.signal_dim,
        fusion_dims=tuple(args.fusion_dims),
        item_id_dim=args.item_id_dim,
        sage_dim=args.sage_dim,
        dropout1=args.dropout1,
        dropout2=args.dropout2,
    )
    model = MetaMMFGCN(dataset.n_users, dataset.m_items, features, config=config_obj, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = vars(args).copy()
    config.update(config_extra)
    config.update({
        "paper": "Liu et al. (2023), Dynamic Multimodal Fusion via Meta-Learning Towards Micro-Video Recommendation, ACM TOIS 42(2)",
        "official_implementation": "https://github.com/iLearn-Lab/TOIS24-MetaMMF",
        "source_fidelity": "Follows official Fusion_model.Net + GCN_model.GCN + BPR path; replaces PyG SAGEConv with equivalent sparse mean GraphSAGE.",
        "feature_preprocessing": "Raw local visual/acoustic/textual arrays are passed to the fusion model, matching the official code's direct np.load feature lookup usage.",
        "mask_training_items": True,
    })

    best_hit20 = -1.0
    best_metrics = None
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        avg_loss = train_one_epoch(model, dataset, mean_adj, optimizer, args, device)
        print(f"Epoch [{epoch:03d}/{args.epochs}] loss={avg_loss:.4f} time={time.time() - start:.2f}s")
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics, eval_users = evaluate(
                model,
                mean_adj,
                aligned_items,
                eval_data,
                device,
                fusion_batch_size=args.fusion_batch_size,
            )
            print(
                f"Evaluation users={eval_users}, aligned_items={len(aligned_items)}, "
                f"Hit@20={metrics.get('Hit@20', 0.0):.4f}, "
                f"NDCG@20={metrics.get('NDCG@20', 0.0):.4f}, "
                f"MRR@20={metrics.get('MRR@20', 0.0):.4f}"
            )
            if metrics.get("Hit@20", 0.0) > best_hit20:
                best_hit20 = metrics["Hit@20"]
                best_metrics = metrics
                os.makedirs(os.path.dirname(args.output), exist_ok=True)
                torch.save({"model_state_dict": model.state_dict(), "config": config, "metrics": best_metrics}, args.output)
                print("Saved best MetaMMF-GCN checkpoint.")

    print_results(best_metrics or {}, config_extra.get("model_name", "MetaMMF-GCN"))

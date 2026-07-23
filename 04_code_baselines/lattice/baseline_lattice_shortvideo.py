#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LATTICE baseline for ShortVideo.

Reference:
    Zhang et al. (2021), "Mining Latent Structures for Multimedia
    Recommendation", ACM MM 2021.

Official implementation:
    https://github.com/CRIPAC-DIG/LATTICE

Implementation notes:
    - This is a ShortVideo adaptation of LATTICE. The official code uses image
      and text features from Amazon datasets; this script uses the local
      ShortVideo visual/acoustic/textual features.
    - The core method follows LATTICE: modality-specific item-item kNN graphs,
      learnable modality fusion, item graph convolution, LightGCN collaborative
      propagation, and pairwise BPR training.
    - Evaluation is restricted to aligned ShortVideo candidate items from
      shortvideo_threepath_aligned_embeddings_1024dim.npz and masks training
      interactions in that same candidate space.
"""

import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import sys
import time

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(ARTIFACT_ROOT, ".."))
INTERACTION_SRC = os.path.join(REPO_ROOT, "Interaction", "src")
CONTENT_DIR = os.path.join(REPO_ROOT, "content")
SHARED_SRC = os.path.join(ARTIFACT_ROOT, "04_code_baselines", "shared_src")

for path in [SHARED_SRC, INTERACTION_SRC, CONTENT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from dataloader import ShortVideoDataset
from model_lattice import LATTICE
from unified_evaluator import print_results


K_LIST = [5, 10, 20]


def parse_args():
    parser = argparse.ArgumentParser(description="LATTICE baseline for ShortVideo.")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(ARTIFACT_ROOT, "01_data_processed", "shortvideo"),
    )
    parser.add_argument(
        "--modal-embeddings",
        default=os.path.join(
            ARTIFACT_ROOT,
            "02_features_embeddings",
            "content_imagebind_embeddings",
            "shortvideo_dualgnn_three_modal_embeddings.npz",
        ),
    )
    parser.add_argument(
        "--aligned-embeddings",
        default=os.path.join(
            ARTIFACT_ROOT,
            "02_features_embeddings",
            "aligned_threepath_embeddings",
            "shortvideo_threepath_aligned_embeddings_1024dim.npz",
        ),
    )
    parser.add_argument("--output", default=os.path.join(SCRIPT_DIR, "best_lattice_shortvideo.pth"))
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--feat-embed-dim", type=int, default=64)
    parser.add_argument("--ui-layers", type=int, default=2)
    parser.add_argument("--item-layers", type=int, default=1)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--lambda-coeff", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--negative-ratio", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2020)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def minibatch(*tensors, batch_size):
    for start in range(0, len(tensors[0]), batch_size):
        yield tuple(t[start:start + batch_size] for t in tensors)


def build_user_item_dict(users, items):
    user_item_dict = {}
    for user, item in zip(users, items):
        user_item_dict.setdefault(int(user), []).append(int(item))
    return user_item_dict


def normalize_rows(matrix):
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)


def load_three_modal_features(path, num_items):
    data = np.load(path, allow_pickle=True)
    required = ["valid_ids", "visual", "acoustic", "textual"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}")

    valid_ids = np.asarray(data["valid_ids"], dtype=np.int64)
    if len(valid_ids) != num_items or not np.array_equal(valid_ids, np.arange(num_items)):
        raise ValueError("ShortVideo LATTICE expects contiguous valid_ids covering all dataset items.")

    visual = normalize_rows(np.asarray(data["visual"], dtype=np.float32))
    acoustic = normalize_rows(np.asarray(data["acoustic"], dtype=np.float32))
    textual = normalize_rows(np.asarray(data["textual"], dtype=np.float32))
    return {"visual": visual, "acoustic": acoustic, "textual": textual}


def load_aligned_item_indices(aligned_embeddings_path, num_items):
    data = np.load(aligned_embeddings_path, allow_pickle=True)
    if "valid_ids" not in data.files:
        raise KeyError(f"{aligned_embeddings_path} must contain 'valid_ids'.")
    valid_ids = np.asarray(data["valid_ids"], dtype=np.int64)
    valid_ids = valid_ids[(valid_ids >= 0) & (valid_ids < num_items)]
    if len(valid_ids) == 0:
        raise ValueError("No aligned ShortVideo item IDs are available.")
    item_id_to_aligned_idx = {int(item_id): idx for idx, item_id in enumerate(valid_ids.tolist())}
    return valid_ids, item_id_to_aligned_idx


def sample_bpr_triplets(dataset, neg_ratio):
    triplets = []
    all_pos = dataset.allPos
    for user, pos_item in zip(dataset.trainUser, dataset.trainItem):
        user = int(user)
        pos_item = int(pos_item)
        user_pos = all_pos[user]
        for _ in range(neg_ratio):
            while True:
                neg_item = np.random.randint(0, dataset.m_items)
                if neg_item not in user_pos:
                    break
            triplets.append((user, pos_item, int(neg_item)))
    return np.asarray(triplets, dtype=np.int64)


def train_one_epoch(model, dataset, optimizer, device, args):
    model.train()
    samples = sample_bpr_triplets(dataset, args.negative_ratio)
    users = torch.LongTensor(samples[:, 0]).to(device)
    pos_items = torch.LongTensor(samples[:, 1]).to(device)
    neg_items = torch.LongTensor(samples[:, 2]).to(device)

    total_loss = 0.0
    total_batches = 0
    build_item_graph = True
    for batch_users, batch_pos, batch_neg in minibatch(
        users, pos_items, neg_items, batch_size=args.batch_size
    ):
        ranking_loss, reg_loss = model.bpr_loss(
            batch_users,
            batch_pos,
            batch_neg,
            build_item_graph=build_item_graph,
        )
        build_item_graph = False
        loss = ranking_loss + args.weight_decay * reg_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().cpu().item()
        total_batches += 1
    return total_loss / max(total_batches, 1)


def build_eval_data(dataset, item_id_to_aligned_idx):
    test_items_dict = build_user_item_dict(dataset.testUser, dataset.testItem)
    train_items_dict = build_user_item_dict(dataset.trainUser, dataset.trainItem)
    eval_users, user_test_items, user_train_items = [], [], []

    for user in sorted(test_items_dict):
        test_filtered = [
            item_id_to_aligned_idx[item]
            for item in test_items_dict[user]
            if item in item_id_to_aligned_idx
        ]
        if not test_filtered:
            continue
        train_filtered = [
            item_id_to_aligned_idx[item]
            for item in train_items_dict.get(user, [])
            if item in item_id_to_aligned_idx
        ]
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


def evaluate(model, dataset, aligned_item_ids, item_id_to_aligned_idx, device):
    model.eval()
    eval_users, user_test_items, user_train_items = build_eval_data(dataset, item_id_to_aligned_idx)
    with torch.no_grad():
        all_users, all_items = model.forward(build_item_graph=True)
        user_indices = torch.LongTensor(eval_users).to(device)
        item_indices = torch.LongTensor(aligned_item_ids).to(device)
        scores = torch.matmul(all_users[user_indices], all_items[item_indices].t())
        for row_idx, train_items in enumerate(user_train_items):
            if train_items:
                scores[row_idx, train_items] = -1e9
        _, top_indices = torch.topk(scores, max(K_LIST), dim=1)
    return compute_metrics_from_topk(top_indices.cpu().numpy(), user_test_items), len(eval_users)


def save_checkpoint(model, path, config, metrics):
    torch.save({"model_state_dict": model.state_dict(), "config": config, "metrics": metrics}, path)


def solve():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 80)
    print("LATTICE Baseline for ShortVideo")
    print("Adapted from ACM MM 2021 LATTICE; visual/acoustic/textual item graph mining")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    dataset = ShortVideoDataset(path=args.data_dir)
    features = load_three_modal_features(args.modal_embeddings, dataset.m_items)
    aligned_item_ids, item_id_to_aligned_idx = load_aligned_item_indices(args.aligned_embeddings, dataset.m_items)
    print(f"Users={dataset.n_users}, Items={dataset.m_items}, aligned_items={len(aligned_item_ids)}")

    model = LATTICE(
        dataset,
        features,
        embedding_dim=args.embedding_dim,
        feat_embed_dim=args.feat_embed_dim,
        ui_layers=args.ui_layers,
        item_layers=args.item_layers,
        topk=args.topk,
        lambda_coeff=args.lambda_coeff,
        device=device,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    config = vars(args).copy()
    config.update({
        "paper": "Zhang et al. (2021), Mining Latent Structures for Multimedia Recommendation, ACM MM 2021",
        "official_implementation": "https://github.com/CRIPAC-DIG/LATTICE",
        "adaptation": "Uses ShortVideo visual/acoustic/textual local embeddings instead of official image/text Amazon features.",
        "candidate_space": "valid_ids from shortvideo_threepath_aligned_embeddings_1024dim.npz",
        "mask_training_items": True,
    })

    best_hit20 = -1.0
    best_metrics = None
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        avg_loss = train_one_epoch(model, dataset, optimizer, device, args)
        print(f"Epoch [{epoch:03d}/{args.epochs}] loss={avg_loss:.4f} time={time.time() - start:.2f}s")
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics, eval_users = evaluate(model, dataset, aligned_item_ids, item_id_to_aligned_idx, device)
            print(
                f"Evaluation users={eval_users}, aligned_items={len(aligned_item_ids)}, "
                f"Hit@20={metrics.get('Hit@20', 0.0):.4f}, "
                f"NDCG@20={metrics.get('NDCG@20', 0.0):.4f}, "
                f"MRR@20={metrics.get('MRR@20', 0.0):.4f}"
            )
            if metrics.get("Hit@20", 0.0) > best_hit20:
                best_hit20 = metrics["Hit@20"]
                best_metrics = metrics
                save_checkpoint(model, args.output, config, best_metrics)
                print("Saved best LATTICE checkpoint.")

    print_results(best_metrics or {}, "LATTICE (ShortVideo)")


if __name__ == "__main__":
    solve()

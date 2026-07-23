#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BPR-MF baseline for ShortVideo.

Reference:
    Rendle et al. (2009), "BPR: Bayesian Personalized Ranking from
    Implicit Feedback", UAI 2009.

Implementation notes:
    - This is an independent matrix factorization baseline, not a LightGCN
      proxy. The model contains only user/item embeddings and dot-product
      scoring.
    - The loss is the standard pairwise BPR objective:
      -log sigmoid(score(u, i_pos) - score(u, i_neg)).
    - ShortVideo experiment baselines in this repository use 1 positive : 8
      negatives for BPR-based training, so NEGATIVE_RATIO is set to 8 for
      consistency with the local experimental protocol.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import sys
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
INTERACTION_SRC = os.path.join(REPO_ROOT, "Interaction", "src")
CONTENT_DIR = os.path.join(REPO_ROOT, "content")

sys.path.append(INTERACTION_SRC)
sys.path.append(CONTENT_DIR)

from dataloader import ShortVideoDataset
from unified_evaluator import RecommendationEvaluator, print_results


EMBEDDING_DIM = 64
EPOCHS = 100
BATCH_SIZE = 2048
LR = 1e-3
WEIGHT_DECAY = 1e-4
NEGATIVE_RATIO = 8
SEED = 2020
EVAL_EVERY = 5
K_LIST = [5, 10, 20]


class BPRMF(nn.Module):
    """Matrix factorization model trained with Bayesian Personalized Ranking."""

    def __init__(self, num_users, num_items, embedding_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def score(self, users, items):
        user_vecs = self.user_embedding(users)
        item_vecs = self.item_embedding(items)
        return torch.sum(user_vecs * item_vecs, dim=1)

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = self.score(users, pos_items)
        neg_scores = self.score(users, neg_items)
        ranking_loss = F.softplus(neg_scores - pos_scores).mean()

        reg_loss = (
            self.user_embedding(users).norm(2).pow(2)
            + self.item_embedding(pos_items).norm(2).pow(2)
            + self.item_embedding(neg_items).norm(2).pow(2)
        ) / (2.0 * users.size(0))
        return ranking_loss, reg_loss


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


def sample_bpr_triplets(dataset, neg_ratio):
    """Sample (user, positive item, negative item) triplets for BPR training."""
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


def train_one_epoch(model, dataset, optimizer, device):
    model.train()
    samples = sample_bpr_triplets(dataset, NEGATIVE_RATIO)

    users = torch.LongTensor(samples[:, 0]).to(device)
    pos_items = torch.LongTensor(samples[:, 1]).to(device)
    neg_items = torch.LongTensor(samples[:, 2]).to(device)

    total_loss = 0.0
    total_batches = 0

    for batch_users, batch_pos, batch_neg in minibatch(
        users, pos_items, neg_items, batch_size=BATCH_SIZE
    ):
        ranking_loss, reg_loss = model.bpr_loss(batch_users, batch_pos, batch_neg)
        loss = ranking_loss + WEIGHT_DECAY * reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().cpu().item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


def load_aligned_item_indices(aligned_embeddings_path, num_items):
    data_npz = np.load(aligned_embeddings_path, allow_pickle=True)
    if "valid_ids" not in data_npz:
        raise KeyError(f"{aligned_embeddings_path} must contain 'valid_ids'.")

    aligned_item_ids = [int(item_id) for item_id in data_npz["valid_ids"]]
    aligned_item_ids = [item_id for item_id in aligned_item_ids if 0 <= item_id < num_items]

    if not aligned_item_ids:
        raise ValueError("No aligned item IDs match the ShortVideo item index space.")

    item_id_to_aligned_idx = {
        item_id: aligned_idx for aligned_idx, item_id in enumerate(aligned_item_ids)
    }
    return aligned_item_ids, item_id_to_aligned_idx


def build_eval_data(dataset, item_id_to_aligned_idx):
    test_items_dict = build_user_item_dict(dataset.testUser, dataset.testItem)
    train_items_dict = build_user_item_dict(dataset.trainUser, dataset.trainItem)

    eval_users = []
    user_test_items = []
    user_train_items = []

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


def evaluate(model, dataset, aligned_embeddings_path, device):
    model.eval()

    aligned_item_ids, item_id_to_aligned_idx = load_aligned_item_indices(
        aligned_embeddings_path, dataset.m_items
    )
    eval_users, user_test_items, user_train_items = build_eval_data(
        dataset, item_id_to_aligned_idx
    )

    if not eval_users:
        raise ValueError("No users have test items in the aligned evaluation space.")

    with torch.no_grad():
        user_indices = torch.LongTensor(eval_users).to(device)
        item_indices = torch.LongTensor(aligned_item_ids).to(device)
        user_embs = model.user_embedding(user_indices)
        item_embs = model.item_embedding(item_indices)

        evaluator = RecommendationEvaluator(device=device)
        metrics = evaluator.evaluate(
            user_embs,
            item_embs,
            user_test_items,
            k_list=K_LIST,
            train_items=user_train_items,
        )

    return metrics, len(eval_users), len(aligned_item_ids)


def save_checkpoint(model, path, config, metrics):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def solve():
    print("=" * 80)
    print("BPR-MF Baseline for ShortVideo")
    print("Setting: 1:8 Negative Sampling, BPR Pairwise Loss, Batch=2048")
    print("=" * 80)

    set_seed(SEED)

    data_dir = os.path.join(REPO_ROOT, "Interaction", "data", "shortvideo")
    aligned_embeddings_path = os.path.join(
        SCRIPT_DIR, "shortvideo_threepath_aligned_embeddings_1024dim.npz"
    )
    ckpt_path = os.path.join(SCRIPT_DIR, "best_bprmf_shortvideo.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset path: {data_dir}")
    print(f"Checkpoint path: {ckpt_path}")

    print("\n[1] Loading Dataset...")
    dataset = ShortVideoDataset(path=data_dir)
    print(f"Users: {dataset.n_users}, Items: {dataset.m_items}")
    print(f"Training interactions: {dataset.trainDataSize}")

    print("\n[2] Initializing BPR-MF...")
    model = BPRMF(dataset.n_users, dataset.m_items, EMBEDDING_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    config = {
        "paper": "Rendle et al. (2009), BPR: Bayesian Personalized Ranking from Implicit Feedback",
        "model": "BPR-MF",
        "embedding_dim": EMBEDDING_DIM,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "negative_ratio": NEGATIVE_RATIO,
        "seed": SEED,
        "evaluation_space": "aligned ShortVideo item IDs from shortvideo_threepath_aligned_embeddings_1024dim.npz",
    }

    print("\n[3] Training...")
    best_hit20 = -1.0
    best_metrics = None

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        avg_loss = train_one_epoch(model, dataset, optimizer, device)
        elapsed = time.time() - start
        print(f"Epoch [{epoch:03d}/{EPOCHS}] loss={avg_loss:.4f} time={elapsed:.2f}s")

        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
            metrics, eval_user_count, aligned_item_count = evaluate(
                model, dataset, aligned_embeddings_path, device
            )
            print(
                f"Evaluation users={eval_user_count}, aligned_items={aligned_item_count}, "
                f"Hit@20={metrics.get('Hit@20', 0.0):.4f}, "
                f"NDCG@20={metrics.get('NDCG@20', 0.0):.4f}, "
                f"MRR@20={metrics.get('MRR@20', 0.0):.4f}"
            )

            if metrics.get("Hit@20", 0.0) > best_hit20:
                best_hit20 = metrics["Hit@20"]
                best_metrics = metrics
                save_checkpoint(model, ckpt_path, config, best_metrics)
                print("Saved best BPR-MF checkpoint.")

    print("\n[4] Final Best Evaluation")
    if best_metrics is None and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_metrics, _, _ = evaluate(model, dataset, aligned_embeddings_path, device)

    print_results(best_metrics or {}, "BPR-MF (ShortVideo)")


if __name__ == "__main__":
    solve()

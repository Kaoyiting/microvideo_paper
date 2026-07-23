#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VBPR baseline for ShortVideo.

Reference:
    He, R., & McAuley, J. (2016), "VBPR: Visual Bayesian Personalized
    Ranking from Implicit Feedback", AAAI 2016.

Public implementation reference:
    Cornac VBPR:
    https://github.com/PreferredAI/cornac/blob/master/cornac/models/vbpr/recom_vbpr.py

Implementation notes:
    - This is a VBPR-style adaptation for the ShortVideo thesis pipeline.
    - The original VBPR paper uses visual CNN features. This project uses the
      existing ImageBind/content item embeddings as fixed visual/content
      features, so results should be reported as an adaptation rather than an
      exact reproduction of the original paper setup.
    - The scoring function follows the VBPR decomposition:
      item bias + collaborative user/item dot product + user visual preference
      dot projected item content feature + global visual bias.
    - Training uses the local baseline protocol: pairwise BPR loss with
      1 positive : 8 negatives. Evaluation is restricted to the same aligned
      ShortVideo candidate item space used by the existing baselines.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import argparse
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
from unified_evaluator import print_results


LATENT_DIM = 64
VISUAL_DIM = 64
EPOCHS = 100
BATCH_SIZE = 2048
LR = 1e-3
WEIGHT_DECAY = 1e-4
NEGATIVE_RATIO = 8
SEED = 2020
EVAL_EVERY = 5
K_LIST = [5, 10, 20]


def parse_args():
    parser = argparse.ArgumentParser(
        description="VBPR baseline for ShortVideo with local ImageBind/content features."
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(ARTIFACT_ROOT, "01_data_processed", "shortvideo"),
        help="ShortVideo dataset directory containing train.txt and test.txt.",
    )
    parser.add_argument(
        "--feature-path",
        default=os.path.join(
            ARTIFACT_ROOT,
            "02_features_embeddings",
            "aligned_threepath_embeddings",
            "shortvideo_threepath_aligned_embeddings_1024dim.npz",
        ),
        help="NPZ file with full item-indexed 'content' embeddings and aligned 'valid_ids'.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(SCRIPT_DIR, "best_vbpr_shortvideo.pth"),
        help="Checkpoint output path.",
    )
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    parser.add_argument("--visual-dim", type=int, default=VISUAL_DIM)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--negative-ratio", type=int, default=NEGATIVE_RATIO)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


class VBPR(nn.Module):
    """Visual Bayesian Personalized Ranking with fixed item content features."""

    def __init__(self, num_users, num_items, feature_matrix, latent_dim, visual_dim):
        super().__init__()
        if feature_matrix.ndim != 2:
            raise ValueError(f"feature_matrix must be 2D, got {feature_matrix.shape}.")
        if feature_matrix.shape[0] != num_items:
            raise ValueError(
                "feature row count must match num_items; "
                f"got rows={feature_matrix.shape[0]}, num_items={num_items}."
            )

        self.num_users = num_users
        self.num_items = num_items
        self.feature_dim = feature_matrix.shape[1]

        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)
        self.user_visual_embedding = nn.Embedding(num_users, visual_dim)
        self.item_bias = nn.Embedding(num_items, 1)
        self.visual_projection = nn.Linear(self.feature_dim, visual_dim, bias=False)
        self.visual_bias = nn.Linear(self.feature_dim, 1, bias=False)

        self.register_buffer("item_features", torch.from_numpy(feature_matrix.astype(np.float32)))

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.normal_(self.user_visual_embedding.weight, std=0.01)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.xavier_uniform_(self.visual_projection.weight)
        nn.init.zeros_(self.visual_bias.weight)

    def score(self, users, items):
        user_vecs = self.user_embedding(users)
        item_vecs = self.item_embedding(items)
        user_visual_vecs = self.user_visual_embedding(users)
        item_visual_vecs = self.visual_projection(self.item_features[items])

        collaborative_score = torch.sum(user_vecs * item_vecs, dim=1)
        visual_score = torch.sum(user_visual_vecs * item_visual_vecs, dim=1)
        bias_score = self.item_bias(items).squeeze(-1)
        visual_bias_score = self.visual_bias(self.item_features[items]).squeeze(-1)
        return collaborative_score + visual_score + bias_score + visual_bias_score

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = self.score(users, pos_items)
        neg_scores = self.score(users, neg_items)
        ranking_loss = F.softplus(neg_scores - pos_scores).mean()

        reg_loss = (
            self.user_embedding(users).norm(2).pow(2)
            + self.item_embedding(pos_items).norm(2).pow(2)
            + self.item_embedding(neg_items).norm(2).pow(2)
            + self.user_visual_embedding(users).norm(2).pow(2)
            + self.item_bias(pos_items).norm(2).pow(2)
            + self.item_bias(neg_items).norm(2).pow(2)
            + self.visual_bias.weight.norm(2).pow(2)
        ) / (2.0 * users.size(0))
        return ranking_loss, reg_loss

    def user_representations(self, users):
        return self.user_embedding(users), self.user_visual_embedding(users)

    def item_representations(self, items):
        return (
            self.item_embedding(items),
            self.visual_projection(self.item_features[items]),
            self.item_bias(items).squeeze(-1),
            self.visual_bias(self.item_features[items]).squeeze(-1),
        )


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
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / (norms + 1e-8)


def load_vbpr_features(feature_path, num_items):
    data_npz = np.load(feature_path, allow_pickle=True)
    if "valid_ids" not in data_npz or "content" not in data_npz:
        raise KeyError(f"{feature_path} must contain 'valid_ids' and 'content'.")

    valid_item_ids = np.asarray(data_npz["valid_ids"], dtype=np.int64)
    content = np.asarray(data_npz["content"], dtype=np.float32)

    if valid_item_ids.ndim != 1:
        raise ValueError(f"valid_ids must be 1D, got shape {valid_item_ids.shape}.")
    if content.ndim != 2:
        raise ValueError(f"content embeddings must be 2D, got shape {content.shape}.")
    if len(valid_item_ids) == 0:
        raise ValueError("valid_ids is empty.")
    if valid_item_ids.min() < 0 or valid_item_ids.max() >= num_items:
        raise ValueError(
            "valid_ids must be dataset item indices within "
            f"[0, {num_items}); got min={valid_item_ids.min()}, "
            f"max={valid_item_ids.max()}."
        )
    if content.shape[0] != num_items:
        raise ValueError(
            "VBPR training expects a full item-indexed content matrix; "
            f"got content rows={content.shape[0]}, dataset items={num_items}."
        )

    return normalize_rows(content), valid_item_ids


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


def train_one_epoch(model, dataset, optimizer, device, batch_size, negative_ratio, weight_decay):
    model.train()
    samples = sample_bpr_triplets(dataset, negative_ratio)

    users = torch.LongTensor(samples[:, 0]).to(device)
    pos_items = torch.LongTensor(samples[:, 1]).to(device)
    neg_items = torch.LongTensor(samples[:, 2]).to(device)

    total_loss = 0.0
    total_batches = 0

    for batch_users, batch_pos, batch_neg in minibatch(
        users, pos_items, neg_items, batch_size=batch_size
    ):
        ranking_loss, reg_loss = model.bpr_loss(batch_users, batch_pos, batch_neg)
        loss = ranking_loss + weight_decay * reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().cpu().item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


def build_eval_data(dataset, valid_item_ids):
    item_id_to_candidate_idx = {
        int(item_id): idx for idx, item_id in enumerate(valid_item_ids.tolist())
    }

    test_items_dict = build_user_item_dict(dataset.testUser, dataset.testItem)
    train_items_dict = build_user_item_dict(dataset.trainUser, dataset.trainItem)

    eval_users = []
    user_test_items = []
    user_train_items = []

    for user in sorted(test_items_dict):
        test_filtered = [
            item_id_to_candidate_idx[item]
            for item in test_items_dict[user]
            if item in item_id_to_candidate_idx
        ]
        if not test_filtered:
            continue

        train_filtered = [
            item_id_to_candidate_idx[item]
            for item in train_items_dict.get(user, [])
            if item in item_id_to_candidate_idx
        ]

        eval_users.append(user)
        user_test_items.append(test_filtered)
        user_train_items.append(train_filtered)

    return eval_users, user_test_items, user_train_items


def compute_metrics_from_topk(top_indices, user_test_items):
    results = {}
    for k in K_LIST:
        results[f"Hit@{k}"] = 0.0
        results[f"NDCG@{k}"] = 0.0
        results[f"MRR@{k}"] = 0.0

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
                if item in gt_set:
                    if first_hit_rank == -1:
                        first_hit_rank = rank + 1

            results[f"Hit@{k}"] += 1.0 if first_hit_rank != -1 else 0.0
            if first_hit_rank != -1:
                results[f"MRR@{k}"] += 1.0 / first_hit_rank

            dcg = 0.0
            for rank, item in enumerate(current_top_k):
                if item in gt_set:
                    dcg += 1.0 / np.log2(rank + 2)
            idcg = sum(
                1.0 / np.log2(rank + 2)
                for rank in range(min(k, len(gt_set)))
            )
            if idcg > 0:
                results[f"NDCG@{k}"] += dcg / idcg

    for key in results:
        results[key] /= max(valid_users, 1)
    return results


def evaluate(model, dataset, valid_item_ids, device):
    model.eval()
    eval_users, user_test_items, user_train_items = build_eval_data(dataset, valid_item_ids)

    if not eval_users:
        raise ValueError("No users have test items in the aligned evaluation space.")

    with torch.no_grad():
        user_indices = torch.LongTensor(eval_users).to(device)
        item_indices = torch.LongTensor(valid_item_ids).to(device)

        user_cf, user_visual = model.user_representations(user_indices)
        item_cf, item_visual, item_bias, visual_bias = model.item_representations(item_indices)

        scores = (
            torch.matmul(user_cf, item_cf.t())
            + torch.matmul(user_visual, item_visual.t())
            + item_bias.unsqueeze(0)
            + visual_bias.unsqueeze(0)
        )

        for user_idx, train_items in enumerate(user_train_items):
            if train_items:
                scores[user_idx, train_items] = -1e9

        _, top_indices = torch.topk(scores, max(K_LIST), dim=1)
        metrics = compute_metrics_from_topk(top_indices.cpu().numpy(), user_test_items)

    return metrics, len(eval_users), len(valid_item_ids)


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
    args = parse_args()

    print("=" * 80)
    print("VBPR Baseline for ShortVideo")
    print(
        "Setting: ImageBind/content features, "
        f"1:{args.negative_ratio} Negative Sampling, BPR Pairwise Loss"
    )
    print("=" * 80)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset path: {args.data_dir}")
    print(f"Feature path: {args.feature_path}")
    print(f"Checkpoint path: {args.output}")

    print("\n[1] Loading Dataset...")
    dataset = ShortVideoDataset(path=args.data_dir)
    print(f"Users: {dataset.n_users}, Items: {dataset.m_items}")
    print(f"Training interactions: {dataset.trainDataSize}")

    print("\n[2] Loading content features...")
    feature_matrix, valid_item_ids = load_vbpr_features(args.feature_path, dataset.m_items)
    print(
        f"Feature matrix: {feature_matrix.shape}, "
        f"aligned candidate items: {len(valid_item_ids)}"
    )

    print("\n[3] Initializing VBPR...")
    model = VBPR(
        dataset.n_users,
        dataset.m_items,
        feature_matrix,
        latent_dim=args.latent_dim,
        visual_dim=args.visual_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    config = {
        "paper": "He and McAuley (2016), VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback",
        "public_implementation_reference": "https://github.com/PreferredAI/cornac/blob/master/cornac/models/vbpr/recom_vbpr.py",
        "model": "VBPR",
        "adaptation": "Uses ImageBind/content item embeddings from the local ShortVideo pipeline instead of original CNN visual features.",
        "scoring": "item_bias + user_item_latent_dot + user_visual_dot_projected_content + global_visual_bias",
        "latent_dim": args.latent_dim,
        "visual_dim": args.visual_dim,
        "content_feature_dim": int(feature_matrix.shape[1]),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "negative_ratio": args.negative_ratio,
        "seed": args.seed,
        "feature_path": args.feature_path,
        "evaluation_space": "aligned ShortVideo item IDs from shortvideo_threepath_aligned_embeddings_1024dim.npz",
    }

    print("\n[4] Training...")
    best_hit20 = -1.0
    best_metrics = None

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        avg_loss = train_one_epoch(
            model,
            dataset,
            optimizer,
            device,
            batch_size=args.batch_size,
            negative_ratio=args.negative_ratio,
            weight_decay=args.weight_decay,
        )
        elapsed = time.time() - start
        print(f"Epoch [{epoch:03d}/{args.epochs}] loss={avg_loss:.4f} time={elapsed:.2f}s")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics, eval_user_count, aligned_item_count = evaluate(
                model, dataset, valid_item_ids, device
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
                save_checkpoint(model, args.output, config, best_metrics)
                print("Saved best VBPR checkpoint.")

    print("\n[5] Final Best Evaluation")
    print_results(best_metrics or {}, "VBPR (ShortVideo)")


if __name__ == "__main__":
    solve()

import argparse
import json
import os
import random
import sys
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Interaction", "src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "content")))

from dataloader import ShortVideoDataset
from model_dualgnn import DualGNN


BASE_DIR = r"c:\Users\kao\microvideo"
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "Interaction", "data", "shortvideo")
DEFAULT_MODAL_EMBEDDINGS = os.path.join(
    BASE_DIR,
    "content",
    "shortvideo_experiment",
    "shortvideo_dualgnn_three_modal_embeddings.npz",
)
DEFAULT_OUTPUT = os.path.join(
    BASE_DIR,
    "content",
    "shortvideo_experiment",
    "best_dualgnn_shortvideo_three_modal.pth",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paper-faithful DualGNN baseline for ShortVideo with three modalities."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--modal-embeddings", default=DEFAULT_MODAL_EMBEDDINGS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--decay", type=float, default=1e-4)
    parser.add_argument("--neg-ratio", type=int, default=8)
    parser.add_argument("--user-top-k", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BPRLoss:
    def __init__(self, recmodel, config):
        self.model = recmodel
        self.weight_decay = config["decay"]
        self.opt = torch.optim.Adam(recmodel.parameters(), lr=config["lr"])

    def stage_one(self, users, pos, neg, user_graph=None, user_weight_matrix=None):
        loss, reg_loss = self.model.bpr_loss(
            users,
            pos,
            neg,
            user_graph=user_graph,
            user_weight_matrix=user_weight_matrix,
        )
        total_loss = loss + self.weight_decay * reg_loss
        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()
        return total_loss.detach().cpu().item()


def minibatch(*tensors, batch_size):
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i : i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i : i + batch_size] for x in tensors)


def uniform_sample(dataset, neg_ratio=1):
    all_pos = dataset.allPos
    samples = []

    for user, pos_item in zip(dataset.trainUser, dataset.trainItem):
        pos_for_user = all_pos[user]
        for _ in range(neg_ratio):
            while True:
                neg_item = np.random.randint(0, dataset.m_items)
                if neg_item not in pos_for_user:
                    break
            samples.append([user, pos_item, neg_item])

    return np.asarray(samples, dtype=np.int64)


def load_three_modal_features(path, dataset):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Three-modal embedding file not found: {path}")

    data = np.load(path, allow_pickle=True)
    required_keys = [
        "valid_ids",
        "visual",
        "acoustic",
        "textual",
        "missing_visual_mask",
        "missing_acoustic_mask",
        "missing_textual_mask",
    ]
    for key in required_keys:
        if key not in data.files:
            raise KeyError(f"Missing key '{key}' in {path}")

    valid_ids = data["valid_ids"].astype(np.int64)
    if not np.array_equal(valid_ids, np.arange(len(valid_ids), dtype=np.int64)):
        raise ValueError("valid_ids must be sorted contiguous mapped item indices.")
    if len(valid_ids) != dataset.m_items:
        raise ValueError(
            f"Embedding item count {len(valid_ids)} does not match dataset.m_items {dataset.m_items}."
        )

    metadata = json.loads(str(data["metadata"])) if "metadata" in data.files else {}
    modality_features = {
        "visual": torch.from_numpy(data["visual"].astype(np.float32)),
        "acoustic": torch.from_numpy(data["acoustic"].astype(np.float32)),
        "textual": torch.from_numpy(data["textual"].astype(np.float32)),
    }
    missing_counts = {
        "visual": int(data["missing_visual_mask"].sum()),
        "acoustic": int(data["missing_acoustic_mask"].sum()),
        "textual": int(data["missing_textual_mask"].sum()),
    }
    return modality_features, missing_counts, metadata


def build_user_cooccurrence_graph(dataset, top_k, device):
    print(f"\n[3] Building user co-occurrence graph (top_k={top_k})...")
    user_item = dataset.UserItemNet.tocsr().astype(np.float32)
    co_mat = user_item.dot(user_item.T).tolil()
    co_mat.setdiag(0)
    co_mat = co_mat.tocsr()
    co_mat.eliminate_zeros()

    user_graph = np.zeros((dataset.n_users, top_k), dtype=np.int64)
    user_weights = np.zeros((dataset.n_users, top_k), dtype=np.float32)

    for user in range(dataset.n_users):
        start, end = co_mat.indptr[user], co_mat.indptr[user + 1]
        neighbors = co_mat.indices[start:end]
        weights = co_mat.data[start:end]

        if len(neighbors) == 0:
            user_graph[user, 0] = user
            user_weights[user, 0] = 1.0
            continue

        if len(neighbors) > top_k:
            top_idx = np.argpartition(weights, -top_k)[-top_k:]
            neighbors = neighbors[top_idx]
            weights = weights[top_idx]

        order = np.argsort(-weights)
        neighbors = neighbors[order]
        weights = weights[order].astype(np.float32)
        weights = weights / (weights.sum() + 1e-8)

        user_graph[user, : len(neighbors)] = neighbors
        user_weights[user, : len(weights)] = weights

    return (
        torch.from_numpy(user_graph).long().to(device),
        torch.from_numpy(user_weights).float().to(device),
    )


def test(dataset, recmodel, device, user_graph, user_weight_matrix, eval_batch_size):
    recmodel.eval()
    max_k = 20
    k_list = [5, 10, 20]
    results = {
        "hit": {k: 0.0 for k in k_list},
        "ndcg": {k: 0.0 for k in k_list},
        "mrr": {k: 0.0 for k in k_list},
    }

    with torch.no_grad():
        test_users = list(dataset.testDict.keys())
        n_test_users = 0
        all_users_emb, all_items_emb, _, _ = recmodel.computer(
            user_graph=user_graph,
            user_weight_matrix=user_weight_matrix,
        )

        for batch_users in minibatch(test_users, batch_size=eval_batch_size):
            batch_users_gpu = torch.LongTensor(batch_users).to(device)
            batch_user_emb = all_users_emb[batch_users_gpu]
            scores = torch.matmul(batch_user_emb, all_items_emb.t())

            for row_idx, user in enumerate(batch_users):
                train_items = dataset.getUserPosItems([user])[0]
                scores[row_idx, train_items] = -float("inf")

            _, topk_indices = torch.topk(scores, max_k)
            topk_indices = topk_indices.cpu().numpy()

            for row_idx, user in enumerate(batch_users):
                gt_items = dataset.testDict.get(user, [])
                if len(gt_items) == 0:
                    continue
                n_test_users += 1
                gt_set = set(gt_items)
                pred_items = topk_indices[row_idx]

                for k in k_list:
                    cut_preds = pred_items[:k]
                    if any(item in gt_set for item in cut_preds):
                        results["hit"][k] += 1.0

                    for rank, item in enumerate(cut_preds):
                        if item in gt_set:
                            results["mrr"][k] += 1.0 / (rank + 1)
                            break

                    dcg = sum(
                        1.0 / np.log2(rank + 2)
                        for rank, item in enumerate(cut_preds)
                        if item in gt_set
                    )
                    idcg = sum(
                        1.0 / np.log2(rank + 2)
                        for rank in range(min(len(gt_items), k))
                    )
                    if idcg > 0:
                        results["ndcg"][k] += dcg / idcg

    if n_test_users == 0:
        raise ValueError("No test users with ground-truth items were found.")

    final_metrics = {}
    for k in k_list:
        final_metrics[f"Hit@{k}"] = results["hit"][k] / n_test_users
        final_metrics[f"NDCG@{k}"] = results["ndcg"][k] / n_test_users
        final_metrics[f"MRR@{k}"] = results["mrr"][k] / n_test_users
    return final_metrics


def print_metrics(metrics):
    print("-" * 65)
    print(f"{'Metric':<10} | {'@5':<15} | {'@10':<15} | {'@20':<15}")
    print("-" * 65)
    for metric_name in ["Hit", "NDCG", "MRR"]:
        print(
            f"{metric_name:<10} | "
            f"{metrics.get(f'{metric_name}@5', 0.0):.4f}          | "
            f"{metrics.get(f'{metric_name}@10', 0.0):.4f}          | "
            f"{metrics.get(f'{metric_name}@20', 0.0):.4f}"
        )
    print("-" * 65)


def solve():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 80)
    print("DualGNN Baseline for ShortVideo")
    print("Paper-faithful setting: visual/acoustic/textual, user co-occurrence graph")
    print(f"Training: BPR Loss, negative sampling 1:{args.neg_ratio}, batch={args.batch_size}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n[1] Loading Dataset...")
    dataset = ShortVideoDataset(path=args.data_dir)
    print(f"    Users: {dataset.n_users}, Items: {dataset.m_items}")

    print("\n[2] Loading three-modal embeddings...")
    modality_features, missing_counts, metadata = load_three_modal_features(
        args.modal_embeddings,
        dataset,
    )
    for name, features in modality_features.items():
        print(f"    {name:<8}: {tuple(features.shape)}, missing={missing_counts[name]}")
    if metadata:
        print(f"    Merge policy: {metadata.get('merge_policy', 'N/A')}")

    user_graph, user_weight_matrix = build_user_cooccurrence_graph(
        dataset,
        top_k=args.user_top_k,
        device=device,
    )

    print("\n[4] Initializing DualGNN...")
    config = {
        "latent_dim_rec": args.latent_dim,
        "lightGCN_n_layers": args.layers,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "decay": args.decay,
        "device": device,
    }
    model = DualGNN(
        config,
        dataset,
        modality_features=modality_features,
    ).to(device)
    loss_func = BPRLoss(model, config)

    print("\n[5] Training Loop...")
    best_hit20 = 0.0
    best_metrics = None
    for epoch in range(config["epochs"]):
        start = time.time()
        model.train()
        samples = uniform_sample(dataset, neg_ratio=args.neg_ratio)
        users = torch.from_numpy(samples[:, 0]).long().to(device)
        pos_items = torch.from_numpy(samples[:, 1]).long().to(device)
        neg_items = torch.from_numpy(samples[:, 2]).long().to(device)

        avg_loss = 0.0
        total_batch = int(np.ceil(len(users) / config["batch_size"]))
        for batch_users, batch_pos, batch_neg in tqdm(
            minibatch(users, pos_items, neg_items, batch_size=config["batch_size"]),
            total=total_batch,
            desc=f"Epoch {epoch + 1}/{config['epochs']}",
            ncols=100,
        ):
            avg_loss += loss_func.stage_one(
                batch_users,
                batch_pos,
                batch_neg,
                user_graph=user_graph,
                user_weight_matrix=user_weight_matrix,
            )
        avg_loss /= max(total_batch, 1)

        print(
            f"EPOCH[{epoch + 1}/{config['epochs']}] "
            f"loss {avg_loss:.4f} time: {time.time() - start:.2f}s"
        )

        should_eval = (epoch + 1) % args.eval_every == 0 or (epoch + 1) == config["epochs"]
        if should_eval:
            print("  Evaluating...")
            metrics = test(
                dataset,
                model,
                device,
                user_graph=user_graph,
                user_weight_matrix=user_weight_matrix,
                eval_batch_size=args.eval_batch_size,
            )
            print_metrics(metrics)
            if metrics["Hit@20"] > best_hit20:
                best_hit20 = metrics["Hit@20"]
                best_metrics = metrics
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "modality_names": model.modality_names,
                        "metrics": metrics,
                    },
                    args.output,
                )
                print(f"  -> Saved best model to {args.output}")

    if best_metrics is not None:
        print("\nBest validation/test metrics by Hit@20:")
        print_metrics(best_metrics)


if __name__ == "__main__":
    solve()

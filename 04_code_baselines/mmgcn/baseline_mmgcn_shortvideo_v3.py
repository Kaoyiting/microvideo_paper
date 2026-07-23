import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import random

import numpy as np
import torch


def set_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
INTERACTION_SRC = os.path.join(REPO_ROOT, "Interaction", "src")
CONTENT_DIR = os.path.join(REPO_ROOT, "content")

sys.path.append(INTERACTION_SRC)
sys.path.append(CONTENT_DIR)

from dataloader import ShortVideoDataset
from model_mmgcn_shortvideo import MMGCN


def minibatch(*tensors, batch_size):
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i : i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i : i + batch_size] for x in tensors)


def sample_bpr_triplets(dataset, neg_ratio=8):
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


class BPRLoss:
    def __init__(self, recmodel, config):
        self.model = recmodel
        self.weight_decay = config["decay"]
        self.opt = torch.optim.Adam(recmodel.parameters(), lr=config["lr"], weight_decay=0.0)

    def stage_one(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        total_loss = loss + self.weight_decay * reg_loss
        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()
        return total_loss.detach().cpu().item()


def load_three_modal_features(path, num_items):
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

    valid_ids = np.asarray(data["valid_ids"], dtype=np.int64)
    if len(valid_ids) == 0:
        raise ValueError("valid_ids is empty.")

    valid_set = set(valid_ids.tolist())
    if min(valid_set) < 0 or max(valid_set) >= num_items:
        raise ValueError("valid_ids contains out-of-range mapped item ids.")

    visual = np.asarray(data["visual"], dtype=np.float32)
    acoustic = np.asarray(data["acoustic"], dtype=np.float32)
    textual = np.asarray(data["textual"], dtype=np.float32)
    if visual.shape[0] != len(valid_ids) or acoustic.shape[0] != len(valid_ids) or textual.shape[0] != len(valid_ids):
        raise ValueError("Modal feature row counts must match valid_ids length.")

    visual_full = np.zeros((num_items, visual.shape[1]), dtype=np.float32)
    acoustic_full = np.zeros((num_items, acoustic.shape[1]), dtype=np.float32)
    textual_full = np.zeros((num_items, textual.shape[1]), dtype=np.float32)
    matched_count = 0
    for row_idx, item_id in enumerate(valid_ids):
        visual_full[item_id] = visual[row_idx]
        acoustic_full[item_id] = acoustic[row_idx]
        textual_full[item_id] = textual[row_idx]
        matched_count += 1

    def normalize(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

    return normalize(visual_full), normalize(acoustic_full), normalize(textual_full), matched_count


def evaluate(dataset, model, device):
    model.eval()
    k_list = [5, 10, 20]
    max_k = max(k_list)
    results = {"hit": {k: 0.0 for k in k_list}, "ndcg": {k: 0.0 for k in k_list}, "mrr": {k: 0.0 for k in k_list}}

    with torch.no_grad():
        test_users = list(dataset.testDict.keys())
        n_test_users = 0
        all_users_emb, all_items_emb = model.forward()
        for batch_users in minibatch(test_users, batch_size=100):
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
                if not gt_items:
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
                    dcg = sum(1.0 / np.log2(rank + 2) for rank, item in enumerate(cut_preds) if item in gt_set)
                    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(gt_items), k)))
                    if idcg > 0:
                        results["ndcg"][k] += dcg / idcg

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
    set_seed(2024)
    print("=" * 80)
    print("MMGCN Baseline for ShortVideo v3")
    print("Original-MMGCN-style three modalities, ShortVideo only, 1:8 negative sampling")
    print("Note: ShortVideo has no separate validation split in this pipeline; periodic evaluation is performed on test.")
    print("=" * 80)

    base_dir = REPO_ROOT
    data_dir = os.path.join(base_dir, "Interaction", "data", "shortvideo")
    modal_path = os.path.join(base_dir, "content", "shortvideo_experiment", "shortvideo_dualgnn_three_modal_embeddings.npz")
    output_path = os.path.join(base_dir, "content", "shortvideo_experiment", "best_mmgcn_shortvideo_v3.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n[1] Loading Dataset...")
    dataset = ShortVideoDataset(path=data_dir)
    print(f"    Users: {dataset.n_users}, Items: {dataset.m_items}")

    print("\n[2] Loading Three-Modal Embeddings...")
    visual, acoustic, textual, matched_count = load_three_modal_features(modal_path, dataset.m_items)
    print(f"    Matched {matched_count}/{dataset.m_items} items with three-modal features.")

    config = {
        "latent_dim_rec": 64,
        "lightGCN_n_layers": 3,
        "keep_prob": 1.0,
        "batch_size": 2048,
        "epochs": 100,
        "lr": 1e-3,
        "decay": 1e-4,
        "device": device,
    }

    print("\n[3] Initializing MMGCN...")
    model = MMGCN(
        config,
        dataset,
        {
            "visual": visual,
            "acoustic": acoustic,
            "textual": textual,
        },
    ).to(device)
    loss_func = BPRLoss(model, config)

    print("\n[4] Training Loop...")
    best_hit20 = 0.0
    for epoch in range(config["epochs"]):
        start = time.time()
        model.train()
        samples = sample_bpr_triplets(dataset, neg_ratio=8)
        users = torch.LongTensor(samples[:, 0]).to(device)
        pos_items = torch.LongTensor(samples[:, 1]).to(device)
        neg_items = torch.LongTensor(samples[:, 2]).to(device)

        total_loss = 0.0
        batch_count = 0
        for batch_users, batch_pos, batch_neg in minibatch(users, pos_items, neg_items, batch_size=config["batch_size"]):
            total_loss += loss_func.stage_one(batch_users, batch_pos, batch_neg)
            batch_count += 1
        avg_loss = total_loss / max(batch_count, 1)

        print(f"EPOCH[{epoch + 1}/{config['epochs']}] loss {avg_loss:.4f} time: {time.time() - start:.2f}s")

        if (epoch + 1) % 5 == 0 or (epoch + 1) == config["epochs"]:
            print("  Evaluating...")
            metrics = evaluate(dataset, model, device)
            print_metrics(metrics)
            if metrics["Hit@20"] > best_hit20:
                best_hit20 = metrics["Hit@20"]
                torch.save(model.state_dict(), output_path)
                print("  -> Saved best model")


if __name__ == "__main__":
    solve()

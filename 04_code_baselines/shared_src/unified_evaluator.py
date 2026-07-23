import numpy as np
import torch


class RecommendationEvaluator:
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def evaluate(self, user_embeddings, item_embeddings, test_items, k_list=None, train_items=None):
        if k_list is None:
            k_list = [5, 10, 20]
        if len(user_embeddings) == 0:
            return {}

        if isinstance(user_embeddings, np.ndarray):
            user_embeddings = torch.FloatTensor(user_embeddings)
        if isinstance(item_embeddings, np.ndarray):
            item_embeddings = torch.FloatTensor(item_embeddings)

        user_embeddings = user_embeddings.to(self.device).contiguous()
        item_embeddings = item_embeddings.to(self.device).contiguous()
        scores = torch.mm(user_embeddings, item_embeddings.t())

        if train_items is not None:
            for user_idx, t_items in enumerate(train_items):
                mask_indices = [idx for idx in t_items if idx < item_embeddings.shape[0]]
                if mask_indices:
                    scores[user_idx, mask_indices] = -1e9

        max_k = max(k_list)
        _, top_indices = torch.topk(scores, max_k, dim=1)
        top_indices = top_indices.cpu().numpy()

        results = {}
        for k in k_list:
            results[f"Hit@{k}"] = 0.0
            results[f"NDCG@{k}"] = 0.0
            results[f"MRR@{k}"] = 0.0

        valid_users = 0
        for user_idx, ground_truth in enumerate(test_items):
            if len(ground_truth) == 0:
                continue
            valid_users += 1
            pred_items = top_indices[user_idx]
            gt_set = set(ground_truth)

            for k in k_list:
                current_top_k = pred_items[:k]
                hit = 0
                first_hit_rank = -1
                dcg = 0.0

                for rank, item in enumerate(current_top_k):
                    if item in gt_set:
                        hit = 1
                        dcg += 1.0 / np.log2(rank + 2)
                        if first_hit_rank == -1:
                            first_hit_rank = rank + 1

                idcg = sum(
                    1.0 / np.log2(rank + 2)
                    for rank in range(min(k, len(gt_set)))
                )

                results[f"Hit@{k}"] += hit
                results[f"NDCG@{k}"] += dcg / idcg if idcg > 0 else 0.0
                if first_hit_rank != -1:
                    results[f"MRR@{k}"] += 1.0 / first_hit_rank

        if valid_users == 0:
            return results

        for key in results:
            results[key] /= valid_users
        return results


def print_results(metrics, model_name="Model"):
    print(f"\n--- {model_name} Evaluation Results ---")
    header = f"{'K':<5} | {'Hit':<10} | {'NDCG (Str)':<12} | {'MRR':<10}"
    print(header)
    print("-" * len(header))
    for k in [5, 10, 20]:
        print(
            f"{k:<5} | {metrics.get(f'Hit@{k}', 0):.4f}     | "
            f"{metrics.get(f'NDCG@{k}', 0):.4f}       | "
            f"{metrics.get(f'MRR@{k}', 0):.4f}"
        )

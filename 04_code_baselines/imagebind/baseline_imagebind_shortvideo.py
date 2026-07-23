import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import torch
import numpy as np
import sys

# Add paths to use existing modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(ARTIFACT_ROOT, ".."))
SHARED_SRC = os.path.join(ARTIFACT_ROOT, "04_code_baselines", "shared_src")
for path in [SHARED_SRC, os.path.join(REPO_ROOT, "Interaction", "src"), os.path.join(REPO_ROOT, "content")]:
    if path not in sys.path:
        sys.path.insert(0, path)
from dataloader import ShortVideoDataset
from unified_evaluator import RecommendationEvaluator, print_results


def build_user_item_dict(users, items):
    user_item_dict = {}
    for user, item in zip(users, items):
        user_item_dict.setdefault(int(user), []).append(int(item))
    return user_item_dict


def load_content_candidate_embeddings(aligned_embeddings_path, num_items):
    data_npz = np.load(aligned_embeddings_path, allow_pickle=True)
    if "valid_ids" not in data_npz or "content" not in data_npz:
        raise KeyError(
            f"{aligned_embeddings_path} must contain 'valid_ids' and 'content'."
        )

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
            "ShortVideo valid_ids must be dataset item indices within "
            f"[0, {num_items}); got min={valid_item_ids.min()}, "
            f"max={valid_item_ids.max()}."
        )

    # This ShortVideo npz stores full item-indexed content rows (dataset item id
    # -> embedding row) while valid_ids stores the aligned evaluation space.
    # Compact the candidate matrix so evaluator labels and embedding rows share
    # the same coordinate system.
    if content.shape[0] == num_items:
        candidate_content = content[valid_item_ids]
    elif content.shape[0] == len(valid_item_ids):
        candidate_content = content
    else:
        raise ValueError(
            "content row count must match either dataset.m_items or len(valid_ids); "
            f"got content rows={content.shape[0]}, dataset items={num_items}, "
            f"valid_ids={len(valid_item_ids)}."
        )

    item_id_to_candidate_idx = {
        int(item_id): idx for idx, item_id in enumerate(valid_item_ids.tolist())
    }
    return candidate_content, item_id_to_candidate_idx, valid_item_ids


def solve():
    print("="*80)
    print("ImageBind (Content-only) Baseline for ShortVideo")
    print("="*80)

    data_dir = os.path.join(ARTIFACT_ROOT, "01_data_processed", "shortvideo")
    aligned_embeddings_path = os.path.join(
        ARTIFACT_ROOT,
        "02_features_embeddings",
        "aligned_threepath_embeddings",
        "shortvideo_threepath_aligned_embeddings_1024dim.npz",
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Dataset
    dataset = ShortVideoDataset(path=data_dir)
    
    # 2. Load aligned item search space.
    content_candidates, item_id_to_candidate_idx, valid_item_ids = (
        load_content_candidate_embeddings(aligned_embeddings_path, dataset.m_items)
    )
    content_embs = torch.FloatTensor(content_candidates).to(device)
    print(
        f"Aligned candidate items: {len(valid_item_ids)} / "
        f"dataset items: {dataset.m_items}"
    )

    # 3. Prepare user histories and test data in dataset item-index space.
    user_history = build_user_item_dict(dataset.trainUser, dataset.trainItem)
    test_items_dict = build_user_item_dict(dataset.testUser, dataset.testItem)

    # 4. Generate user embeddings from the latest training item with a content
    # embedding, then map test/train items into the compact candidate space.
    user_embs = []
    user_test_items = []
    user_train_items = []

    for user in sorted(test_items_dict):
        history = [
            item_id_to_candidate_idx[item]
            for item in user_history.get(user, [])
            if item in item_id_to_candidate_idx
        ]
        if not history:
            continue

        test_filtered = [
            item_id_to_candidate_idx[item]
            for item in test_items_dict[user]
            if item in item_id_to_candidate_idx
        ]
        if not test_filtered:
            continue

        user_embs.append(content_embs[history[-1]])
        user_test_items.append(test_filtered)
        user_train_items.append(history)

    if not user_embs:
        print("Error: No test users with history and content embeddings found.")
        return

    user_embs = torch.stack(user_embs)
    print(f"Evaluating on {len(user_test_items)} users with valid content history/test items.")

    # 5. Evaluate
    evaluator = RecommendationEvaluator(device=device)
    metrics = evaluator.evaluate(
        user_embs,
        content_embs,
        user_test_items,
        k_list=[5, 10, 20],
        train_items=user_train_items,
    )
    
    print_results(metrics, "ImageBind (Content-only)")

if __name__ == "__main__":
    solve()

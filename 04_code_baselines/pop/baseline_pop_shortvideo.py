import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SHARED_SRC = os.path.join(ARTIFACT_ROOT, "04_code_baselines", "shared_src")
if SHARED_SRC not in sys.path:
    sys.path.insert(0, SHARED_SRC)
from unified_evaluator import RecommendationEvaluator, print_results

def solve():
    print("="*80)
    print("POP Baseline for ShortVideo")
    print("="*80)
    
    data_dir = os.path.join(ARTIFACT_ROOT, "01_data_processed", "shortvideo")
    train_file = os.path.join(data_dir, "train.txt")
    test_file = os.path.join(data_dir, "test.txt")
    aligned_embeddings_path = os.path.join(
        ARTIFACT_ROOT,
        "02_features_embeddings",
        "aligned_threepath_embeddings",
        "shortvideo_threepath_aligned_embeddings_1024dim.npz",
    )

    # 1. Load Aligned Item Search Space
    print("Loading aligned item search space...")
    data_npz = np.load(aligned_embeddings_path, allow_pickle=True)
    aligned_item_ids = [str(tid) for tid in data_npz['valid_ids']]
    item_id_to_aligned_idx = {tid: idx for idx, tid in enumerate(aligned_item_ids)}

    # 2. Count Popularity and extract train items for masking
    print("Counting popularity...")
    item_counts = {}
    train_dict = {}
    max_item_id = 0
    with open(train_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                u, item = int(parts[0]), int(parts[1])
                item_counts[item] = item_counts.get(item, 0) + 1
                max_item_id = max(max_item_id, item)
                
                if u not in train_dict: train_dict[u] = []
                train_dict[u].append(item)
    
    # 2. Sort Items
    sorted_items = sorted(item_counts.keys(), key=lambda x: item_counts[x], reverse=True)
    
    # 3. Load Test Data
    print("Loading test data...")
    test_dict = {}
    with open(test_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                u, i = int(parts[0]), int(parts[1])
                if u not in test_dict: test_dict[u] = []
                test_dict[u].append(i)
                
    sorted_users = sorted(test_dict.keys())
    user_test_items = [test_dict[u] for u in sorted_users]
    user_train_items = [train_dict.get(u, []) for u in sorted_users]
    
    # 4. Evaluate
    print("Evaluating...")
    # Map popularity to scores for RecommendationEvaluator
    # We only care about scores for items in the aligned search space
    item_scores = np.zeros(len(aligned_item_ids))
    for i, tid in enumerate(aligned_item_ids):
        item_id = int(tid)
        item_scores[i] = item_counts.get(item_id, 0)
        
    # Prepare test data
    filtered_user_test_items = []
    filtered_user_train_items = []
    
    for u in sorted_users:
        test_ids = [str(i) for i in test_dict[u]]
        test_filtered = [item_id_to_aligned_idx[tid] for tid in test_ids if tid in item_id_to_aligned_idx]
        
        if len(test_filtered) > 0:
            filtered_user_test_items.append(test_filtered)
            
            train_ids = [str(i) for i in train_dict.get(u, [])]
            train_filtered = [item_id_to_aligned_idx[tid] for tid in train_ids if tid in item_id_to_aligned_idx]
            filtered_user_train_items.append(train_filtered)
            
    user_embs = torch.ones(len(filtered_user_test_items), 1)
    item_embs = torch.FloatTensor(item_scores).view(-1, 1)
    
    evaluator = RecommendationEvaluator(device='cpu')
    metrics = evaluator.evaluate(user_embs, item_embs, filtered_user_test_items, k_list=[5, 10, 20], train_items=filtered_user_train_items)
    
    print_results(metrics, "POP (ShortVideo)")

if __name__ == "__main__":
    solve()

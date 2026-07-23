#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 9: Evaluate Three-Path Recommendation Performance (ShortVideo)

Compare different fusion strategies:
- Strategy A: Semantic + Interaction (two-path baseline)
- Strategy B: Semantic + Interaction + Content (three-path, equal weight)
- Strategy C: Semantic + Interaction + Content (three-path, optimized weights)

Metrics: Hit@K, NDCG@K, MRR@K for K=[5, 10, 20]
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random

# ============ Configuration ============
script_dir = os.path.dirname(os.path.abspath(__file__))
LIGHTGCN_CHECKPOINT_PATH = os.path.abspath(os.path.join(script_dir, "../../Interaction/src/best_lightsasrec.pth"))
ALIGNED_EMBEDDINGS_PATH = os.path.join(script_dir, "shortvideo_threepath_aligned_embeddings_1024dim.npz")
ALIGNMENT_MODEL_PATH = os.path.join(script_dir, "best_shortvideo_threepath_alignment.pth")
TRAIN_DATA_PATH = os.path.abspath(os.path.join(script_dir, "../../Interaction/data/shortvideo/train.txt"))
TEST_DATA_PATH = os.path.abspath(os.path.join(script_dir, "../../Interaction/data/shortvideo/test.txt"))
DATA_INDEX_PATH = os.path.join(script_dir, "data_index_shortvideo.json")

K_VALUES = [5, 10, 20]
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED = 2024

# ============ Model Definition ============

class ThreePathAlignmentModel(nn.Module):
    """Three-path alignment model"""
    def __init__(self, semantic_dim=1536, interaction_dim=64, content_dim=1024, output_dim=1024):
        super().__init__()
        
        # Semantic path: 1536 -> 1024
        self.semantic_proj = nn.Sequential(
            nn.Linear(semantic_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Interaction path: 64 -> 1024
        self.interaction_proj = nn.Sequential(
            nn.Linear(interaction_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Content path: 1024 -> 1024
        self.content_proj = nn.Sequential(
            nn.Linear(content_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, semantic=None, interaction=None, content=None):
        outputs = {}
        
        if semantic is not None:
            outputs['semantic'] = F.normalize(self.semantic_proj(semantic), dim=-1)
        
        if interaction is not None:
            outputs['interaction'] = F.normalize(self.interaction_proj(interaction), dim=-1)
        
        if content is not None:
            outputs['content'] = F.normalize(self.content_proj(content), dim=-1)
        
        return outputs

# ============ Helper Functions ============

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_train_test_data():
    """
    Load train and test data
    Returns: train_dict, test_dict (user_id -> list of item_ids)
    """
    train_dict = {}
    test_dict = {}
    
    # Load train data (format: user_id item_id per line)
    with open(TRAIN_DATA_PATH, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                user_id = int(parts[0])
                item_id = int(parts[1])
                if user_id not in train_dict:
                    train_dict[user_id] = []
                train_dict[user_id].append(item_id)
    
    # Load test data (format: user_id item_id per line)
    with open(TEST_DATA_PATH, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                user_id = int(parts[0])
                item_id = int(parts[1])
                if user_id not in test_dict:
                    test_dict[user_id] = []
                test_dict[user_id].append(item_id)
    
    return train_dict, test_dict

def compute_metrics_gpu(user_embeddings, item_embeddings, train_dict, test_dict, k_values, device):
    """
    GPU-accelerated metric computation using pre-trained user embeddings
    Args:
        user_embeddings: (n_users, D) array of user embeddings
        item_embeddings: (n_items, D) array of item embeddings
        train_dict: user_id -> list of training items
        test_dict: user_id -> list of test items
    Returns: dict of metrics
    """
    # Convert to GPU tensors
    user_embeddings = torch.FloatTensor(user_embeddings).to(device)
    item_embeddings = torch.FloatTensor(item_embeddings).to(device)
    
    # Normalize embeddings
    user_embeddings = F.normalize(user_embeddings, dim=1)
    item_embeddings = F.normalize(item_embeddings, dim=1)
    
    # Compute similarities: [n_users, n_items]
    similarities = torch.mm(user_embeddings, item_embeddings.t())
    
    # Mask out training items
    # CRITICAL FIX: Ensure user_ids matches the order of user_embeddings (which was sorted)
    test_user_ids = sorted(list(test_dict.keys()))
    for user_idx, user_id in enumerate(test_user_ids):
        train_items = train_dict.get(user_id, [])
        if len(train_items) > 0:
            similarities[user_idx, train_items] = -float('inf')
    
    # Get top-K items for each user
    max_k = max(k_values)
    _, top_indices = torch.topk(similarities, max_k, dim=1)
    top_indices = top_indices.cpu().numpy()
    
    # Calculate metrics
    metrics = {f'{metric}@{k}': [] for metric in ['Hit', 'NDCG', 'MRR'] for k in k_values}
    
    for user_idx, user_id in enumerate(test_user_ids):
        user_test_items = test_dict[user_id]
        
        if len(user_test_items) == 0:
            continue
        
        pred_items = top_indices[user_idx]
        
        for k in k_values:
            pred_k = pred_items[:k]
            
            # Hit@K
            hit = 1.0 if len(set(pred_k) & set(user_test_items)) > 0 else 0.0
            metrics[f'Hit@{k}'].append(hit)
            
            # NDCG@K
            dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(pred_k) 
                     if item in user_test_items)
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(user_test_items), k)))
            ndcg = dcg / idcg if idcg > 0 else 0
            metrics[f'NDCG@{k}'].append(ndcg)
            
            # MRR@K
            mrr = 0
            for i, item in enumerate(pred_k):
                if item in user_test_items:
                    mrr = 1 / (i + 1)
                    break
            metrics[f'MRR@{k}'].append(mrr)
    
    # Average metrics
    result = {}
    for key, values in metrics.items():
        result[key] = np.mean(values) if len(values) > 0 else 0.0
    
    return result

# ============ Evaluation Strategies ============

def evaluate_strategy_a(aligned_data, user_embeddings, train_dict, test_dict, device):
    """
    Strategy A: Semantic + Interaction (two-path baseline)
    """
    print("\n[Strategy A] Semantic + Interaction (Two-Path)")
    
    semantic_embs = aligned_data['semantic']
    interaction_embs = aligned_data['interaction']
    
    # Fusion: average of semantic and interaction
    item_embeddings = (semantic_embs + interaction_embs) / 2.0
    
    # Compute metrics using pre-trained user embeddings
    metrics = compute_metrics_gpu(user_embeddings, item_embeddings, train_dict, test_dict, K_VALUES, device)
    
    return metrics

def evaluate_strategy_b(aligned_data, user_embeddings, train_dict, test_dict, device):
    """
    Strategy B: Semantic + Interaction + Content (equal weight)
    """
    print("\n[Strategy B] Semantic + Interaction + Content (Equal Weight)")
    
    semantic_embs = aligned_data['semantic']
    interaction_embs = aligned_data['interaction']
    content_embs = aligned_data['content']
    
    # Fusion: average of all three
    item_embeddings = (semantic_embs + interaction_embs + content_embs) / 3.0
    
    # Compute metrics using pre-trained user embeddings
    metrics = compute_metrics_gpu(user_embeddings, item_embeddings, train_dict, test_dict, K_VALUES, device)
    
    return metrics

def evaluate_strategy_c(aligned_data, user_embeddings, train_dict, test_dict, device, 
                       weights=(0.3, 0.3, 0.4)):
    """
    Strategy C: Semantic + Interaction + Content (weighted fusion)
    Default: semantic=0.3, interaction=0.3, content=0.4
    """
    w_sem, w_int, w_con = weights
    print(f"\n[Strategy C] Semantic + Interaction + Content (Weighted: {w_sem}/{w_int}/{w_con})")
    
    semantic_embs = aligned_data['semantic']
    interaction_embs = aligned_data['interaction']
    content_embs = aligned_data['content']
    
    # Weighted fusion
    item_embeddings = (w_sem * semantic_embs + 
                      w_int * interaction_embs + 
                      w_con * content_embs)
    
    # Compute metrics using pre-trained user embeddings
    metrics = compute_metrics_gpu(user_embeddings, item_embeddings, train_dict, test_dict, K_VALUES, device)
    
    return metrics

# ============ Main Evaluation ============

def main():
    print("="*80)
    print("Phase 9: Evaluate Three-Path Recommendation Performance (ShortVideo)")
    print("="*80)
    
    set_seed(SEED)
    
    # [1] Load data
    print("\n[1/5] 載入資料...")
    
    print("  [1.1] 載入訓練/測試資料...")
    train_dict, test_dict = load_train_test_data()
    print(f"        訓練用戶數: {len(train_dict)}")
    print(f"        測試用戶數: {len(test_dict)}")
    
    print("  [1.2] 載入對齊 embeddings...")
    aligned_data = np.load(ALIGNED_EMBEDDINGS_PATH)
    
    semantic_embs = aligned_data['semantic']
    interaction_embs = aligned_data['interaction']
    content_embs = aligned_data['content']
    valid_ids = aligned_data['valid_ids']
    
    print(f"        Semantic shape: {semantic_embs.shape}")
    print(f"        Interaction shape: {interaction_embs.shape}")
    print(f"        Content shape: {content_embs.shape}")
    print(f"        有效項目數: {len(valid_ids)}")
    
    print("  [1.3] 載入 LightGCN user embeddings...")
    checkpoint = torch.load(LIGHTGCN_CHECKPOINT_PATH, map_location='cpu')
    user_embeddings_64d = checkpoint['lightgcn.embedding_user.weight'].numpy()
    print(f"        User embeddings (64-dim) shape: {user_embeddings_64d.shape}")
    
    # Filter user embeddings for test users only
    test_user_ids = sorted(list(test_dict.keys()))
    user_embeddings_64d = user_embeddings_64d[test_user_ids]
    print(f"        測試用戶 embeddings (64-dim) shape: {user_embeddings_64d.shape}")
    
    # [2] Setup device
    print(f"\n[2/5] 設置計算設備...")
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"        使用設備: {device}")
    
    # [2.5] Load alignment model and project user embeddings to 1024-dim
    print(f"\n[2.5/5] 投影 user embeddings 到 1024 維度...")
    alignment_model = ThreePathAlignmentModel()
    alignment_checkpoint = torch.load(ALIGNMENT_MODEL_PATH, map_location='cpu')
    alignment_model.load_state_dict(alignment_checkpoint['model_state_dict'])
    alignment_model.to(device)
    alignment_model.eval()
    
    # Project user embeddings through interaction pathway
    with torch.no_grad():
        user_embeddings_tensor = torch.FloatTensor(user_embeddings_64d).to(device)
        user_embeddings_1024d = alignment_model.interaction_proj(user_embeddings_tensor)
        user_embeddings_1024d = F.normalize(user_embeddings_1024d, dim=-1)
        user_embeddings = user_embeddings_1024d.cpu().numpy()
    
    print(f"        User embeddings (1024-dim) shape: {user_embeddings.shape}")
    
    # [3] Evaluate strategies
    print(f"\n[3/5] 評估不同融合策略...")
    
    results = {}
    
    # Strategy A: Semantic + Interaction
    results['Strategy_A'] = evaluate_strategy_a(aligned_data, user_embeddings, train_dict, test_dict, device)
    
    # Strategy B: Equal weight (1/3, 1/3, 1/3)
    results['Strategy_B'] = evaluate_strategy_b(aligned_data, user_embeddings, train_dict, test_dict, device)
    
    # Strategy C: Weighted (0.3, 0.3, 0.4) - emphasize content
    results['Strategy_C'] = evaluate_strategy_c(aligned_data, user_embeddings, train_dict, test_dict, device,
                                                weights=(0.3, 0.3, 0.4))
    
    # [4] Print results
    print(f"\n[4/5] 結果比較...")
    print("\n" + "="*80)
    print("評估結果")
    print("="*80)
    
    # Table header
    print(f"\n{'Strategy':<25}", end="")
    for k in K_VALUES:
        print(f"Hit@{k:<3} NDCG@{k:<3} MRR@{k:<3}  ", end="")
    print()
    print("-" * 80)
    
    # Print each strategy
    strategy_names = {
        'Strategy_A': 'A: Semantic+Interaction',
        'Strategy_B': 'B: Three-Path (Equal)',
        'Strategy_C': 'C: Three-Path (0.3/0.3/0.4)'
    }
    
    for strategy_key, strategy_name in strategy_names.items():
        metrics = results[strategy_key]
        print(f"{strategy_name:<25}", end="")
        
        for k in K_VALUES:
            hit = metrics[f'Hit@{k}']
            ndcg = metrics[f'NDCG@{k}']
            mrr = metrics[f'MRR@{k}']
            print(f"{hit:.4f} {ndcg:.4f} {mrr:.4f}  ", end="")
        print()
    
    # [5] Calculate improvements
    print(f"\n[5/5] 改進分析...")
    print("\n" + "="*80)
    print("相對於 Strategy A 的改進 (%)")
    print("="*80)
    
    baseline = results['Strategy_A']
    
    for strategy_key in ['Strategy_B', 'Strategy_C']:
        metrics = results[strategy_key]
        strategy_name = strategy_names[strategy_key]
        
        print(f"\n{strategy_name}:")
        for k in K_VALUES:
            for metric in ['Hit', 'NDCG', 'MRR']:
                key = f'{metric}@{k}'
                baseline_val = baseline[key]
                current_val = metrics[key]
                
                if baseline_val > 0:
                    improvement = ((current_val - baseline_val) / baseline_val) * 100
                    symbol = "↑" if improvement > 0 else "↓"
                    print(f"  {key:<10}: {improvement:+.2f}% {symbol}")
                else:
                    print(f"  {key:<10}: N/A")
    
    # Summary
    print("\n" + "="*80)
    print("Phase 9 完成！")
    print("="*80)
    print("\n✓ 評估完成，結果已顯示")
    print("✓ 建議策略：根據 Hit@10 和 NDCG@10 選擇最佳策略")
    print("="*80)

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 18: ShortVideo Ultimate Evaluation (Full Rank Fusion)
Purpose: Eliminate the Top-200 retrieval bottleneck of the Rerank strategy.
Usage: Uses LightSASRec (Interaction) + Aligned S/C Embeddings to search the FULL item space.
Logic: Combined Score = 0.8 * I + 0.1 * S + 0.1 * C (Global)
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
interaction_src = os.path.abspath(os.path.join(script_dir, "../../Interaction/src"))
if interaction_src not in sys.path:
    sys.path.insert(0, interaction_src)

from model import LightSASRec
from dataloader import ShortVideoDataset

# ============ Configuration ============
CHECKPOINT_PATH = os.path.abspath(os.path.join(script_dir, "../../Interaction/src/best_lightsasrec.pth"))
DATA_PATH = os.path.abspath(os.path.join(script_dir, "../../Interaction/data/shortvideo"))
ALIGNED_EMBEDDINGS_PATH = os.path.join(script_dir, "shortvideo_threepath_aligned_embeddings_1024dim.npz")

K_VALUES = [5, 10, 20]
MAXLEN = 50
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def compute_metrics(top_k_indices, ground_truth, k_values=[5, 10, 20]):
    metrics = {f'{m}@{k}': [] for m in ['Hit', 'NDCG', 'MRR'] for k in k_values}
    
    for pred_items, true_items in zip(top_k_indices, ground_truth):
        # Ensure true_items is a list/set
        if not isinstance(true_items, (list, set, tuple)):
            true_items = [true_items]
        
        # Ground truth set for fast lookup
        gt_set = set(true_items)
            
        for k in k_values:
            pred_k = pred_items[:k]
            
            # Hit
            hit = 1.0 if len(set(pred_k) & gt_set) > 0 else 0.0
            metrics[f'Hit@{k}'].append(hit)
            
            # NDCG (Strict)
            dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(pred_k) if item in gt_set)
            
            # IDCG (Strict) - Best possible ordering of ground truth items
            num_relevant = len(gt_set)
            k_relevant = min(num_relevant, k)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(k_relevant))
            
            ndcg = dcg / idcg if idcg > 0 else 0.0
            metrics[f'NDCG@{k}'].append(ndcg)
            
            # MRR
            mrr = 0.0
            for i, item in enumerate(pred_k):
                if item in gt_set:
                    mrr = 1.0 / (i + 1)
                    break 
            metrics[f'MRR@{k}'].append(mrr)
            
    return {k: np.mean(v) for k, v in metrics.items()}

def evaluate_full_rank(w1, w2, w3, u_I, u_S, u_C, item_I, item_S, item_C, mask, gt_list):
    # Calculate Full Scores (N_users, N_items)
    # Score = w1 * (u_I @ i_I.T) + w2 * (u_S @ i_S.T) + w3 * (u_C @ i_C.T)
    # Note: u_I, u_S, u_C are already normalized? 
    # Yes, we will normalize them before passing here.
    
    # Combined user/item weights to avoid large memory allocation of 3 separate score matrices
    # Score = (w1*u_I) @ i_I.T + ...
    # Wait, dimensions match? u_I and i_I have same dim? Yes (64 for I, 1024 for S/C)
    
    # We can compute sub-scores and add them? 
    # ShortVideo: 3351 users * 5907 items. ~20M floats. Tiny.
    
    # 1. Interaction Score
    score = w1 * torch.mm(u_I, item_I.t())
    
    # 2. Semantic Score
    if w2 > 0:
        score += w2 * torch.mm(u_S, item_S.t())
        
    # 3. Content Score
    if w3 > 0:
        score += w3 * torch.mm(u_C, item_C.t())
        
    # Apply Mask (Training items)
    if mask is not None:
        score.masked_fill_(mask, -1e9)
        
    # Retrieve top K
    _, top_k = torch.topk(score, max(K_VALUES), dim=1)
    
    return compute_metrics(top_k.cpu().numpy().tolist(), gt_list, K_VALUES)

def main():
    device = torch.device(DEVICE)
    print(f"Device: {device}")
    
    # 1. Load Dataset
    print("Loading ShortVideo dataset...")
    dataset = ShortVideoDataset(path=DATA_PATH)
    config = {'latent_dim_rec': 64, 'lightGCN_n_layers': 3, 'dropout': 0, 'keep_prob': 0.6}
    
    class SASRecArgs:
        def __init__(self):
            self.hidden_units = 64; self.num_blocks = 2; self.num_heads = 1
            self.dropout_rate = 0.2; self.maxlen = MAXLEN; self.device = DEVICE; self.norm_first = True

    model = LightSASRec(config, SASRecArgs(), dataset).to(device)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        print(f"✓ Loaded Base Model: {os.path.basename(CHECKPOINT_PATH)}")
    model.eval()
    
    # 2. Load Aligned Features
    align_data = np.load(ALIGNED_EMBEDDINGS_PATH)
    item_S = F.normalize(torch.FloatTensor(align_data['semantic']).to(device), dim=-1)
    item_C = F.normalize(torch.FloatTensor(align_data['content']).to(device), dim=-1)
    
    # 3. Prepare Test Users and Ground Truth
    test_users = sorted(dataset.testDict.keys())
    gt_list = [dataset.testDict[u] for u in test_users]
    
    # 4. Compute User Embeddings and Training Mask
    print("Computing User Embeddings...")
    _, lightgcn_items = model.lightgcn.computer()
    padding = torch.zeros(1, lightgcn_items.shape[1], device=device)
    full_item_embeddings = torch.cat([padding, lightgcn_items], dim=0) # Index 0 is padding
    
    # Item Interaction Embedding (1..N) to match Item S/C (0..N-1)
    # lightgcn_items[0] corresponds to item ID 0.
    item_I = F.normalize(lightgcn_items, dim=-1)
    
    u_I_list, u_S_list, u_C_list = [], [], []
    mask_list = []
    
    with torch.no_grad():
        for u in tqdm(test_users):
            train_items = dataset.allPos[u]
            
            # --- Interaction (SASRec) ---
            input_seq = [x + 1 for x in train_items[-MAXLEN:]]
            seq = np.zeros(MAXLEN, dtype=np.int32)
            seq[MAXLEN-len(input_seq):] = input_seq
            
            log_feats = model.sasrec.log2feats(torch.LongTensor(seq).unsqueeze(0).to(device), item_emb_matrix=full_item_embeddings)
            u_I_list.append(F.normalize(log_feats[:, -1, :], dim=-1))
            
            # --- Semantic & Content (Mean Pooling) ---
            # items are 0-indexed
            u_S_list.append(item_S[train_items].mean(0, keepdim=True))
            u_C_list.append(item_C[train_items].mean(0, keepdim=True))
            
            # --- Mask ---
            # Create a boolean row for this user. 1 = Mask (Training), 0 = Keep
            # Mask all training items
            m = torch.zeros(dataset.m_items, dtype=torch.bool, device=device)
            m[train_items] = True
            mask_list.append(m)
            
    # Stack Users
    u_I = torch.cat(u_I_list, dim=0)
    u_S = torch.cat(u_S_list, dim=0)
    u_C = torch.cat(u_C_list, dim=0)
    mask = torch.stack(mask_list, dim=0)
    
    u_S = F.normalize(u_S, dim=-1)
    u_C = F.normalize(u_C, dim=-1)
    
    print(f"User Embeddings: {u_I.shape}")
    print(f"Item Embeddings: {item_I.shape}")
    
    # 5. Evaluate Baseline (Interaction Only)
    print("\nEvaluating Interaction (Base)...")
    base_res = evaluate_full_rank(1.0, 0.0, 0.0, u_I, u_S, u_C, item_I, item_S, item_C, mask, gt_list)
    
    # 6. Evaluate Full Fusion (0.8, 0.1, 0.1) as per Phase 17
    print("Evaluating Full Fusion (0.8, 0.1, 0.1)...")
    fusion_res = evaluate_full_rank(0.8, 0.1, 0.1, u_I, u_S, u_C, item_I, item_S, item_C, mask, gt_list)
    
    # 7. Report
    print("\n" + "="*85)
    print(f"{'Method':<25} | {'K':<5} | {'Hit':<10} | {'NDCG (Str)':<12} | {'MRR':<10}")
    print("-" * 85)
    
    def print_rows(name, res):
        for k in K_VALUES:
            print(f"{name:<25} | {k:<5} | {res[f'Hit@{k}']:.4f}     | {res[f'NDCG@{k}']:.4f}       | {res[f'MRR@{k}']:.4f}")

    print_rows("Interaction (Base)", base_res)
    print("-" * 85)
    print_rows("Full Rank (Fusion)", fusion_res)
    print("="*85)
    
    # Compare Hit@10
    base_h10 = base_res['Hit@10']
    fusion_h10 = fusion_res['Hit@10']
    gain = (fusion_h10 - base_h10) / base_h10 * 100
    print(f"Full Rank Fusion Gain (Hit@10): {gain:+.2f}%")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 17 Ultimate: ShortVideo Evaluation (Dot Product + Hidden State)
This script implements the 'Ultimate' re-ranking logic:
1. No L2 Normalization (Preserves Magnitude/Popularity).
2. Uses SASRec Final Hidden State as the User Query for Semantic/Content.
3. Scale-aligned weighted fusion (0.8/0.1/0.1 after scale matching).
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
RECALL_SIZE = 200 
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def compute_metrics(pred_items, true_items, k):
    pred_k = pred_items[:k]
    hit = 1.0 if len(set(pred_k) & set(true_items)) > 0 else 0.0
    dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(pred_k) if item in true_items)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(true_items), k)))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    mrr = 0.0
    for i, item in enumerate(pred_k):
        if item in true_items:
            mrr = 1.0 / (i + 1); break
    return hit, ndcg, mrr

def _min_max_normalize(scores):
    """
    Min-Max normalization to align scales [0, 1] across different modalities.
    """
    min_v = scores.min(dim=1, keepdim=True)[0]
    max_v = scores.max(dim=1, keepdim=True)[0]
    return (scores - min_v) / (max_v - min_v + 1e-8)

def z_score_normalize(scores):
    """
    Min-Max normalization to align scales [0, 1] across different modalities.
    """
    return _min_max_normalize(scores)

import argparse

def main():
    parser = argparse.ArgumentParser(description="Ultimate Ablation Study")
    parser.add_argument("--w_i", type=float, default=0.8, help="Weight for Interaction")
    parser.add_argument("--w_s", type=float, default=0.1, help="Weight for Semantic")
    parser.add_argument("--w_c", type=float, default=0.1, help="Weight for Content")
    args = parser.parse_args()

    w_i, w_s, w_c = args.w_i, args.w_s, args.w_c

    device = torch.device(DEVICE)
    dataset = ShortVideoDataset(path=DATA_PATH)
    config = {'latent_dim_rec': 64, 'lightGCN_n_layers': 3, 'dropout': 0, 'keep_prob': 0.6}
    
    class SASRecArgs:
        def __init__(self):
            self.hidden_units = 64; self.num_blocks = 2; self.num_heads = 1
            self.dropout_rate = 0.2; self.maxlen = MAXLEN; self.device = DEVICE; self.norm_first = True

    model = LightSASRec(config, SASRecArgs(), dataset).to(device)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        print(f"[OK] Loaded Base Model: {os.path.basename(CHECKPOINT_PATH)}")
    model.eval()
    
    # Load Aligned Features - NO NORMALIZATION
    align_data = np.load(ALIGNED_EMBEDDINGS_PATH)
    item_S = torch.FloatTensor(align_data['semantic']).to(device)
    item_C = torch.FloatTensor(align_data['content']).to(device)
    
    # Project SASRec Hidden State to 1024-dim space if needed, 
    # but here we use the Mean of history aligned embeddings as user query 
    # for S and C to avoid needing the Phase 8 projection layers.
    
    test_users = sorted(dataset.testDict.keys())
    
    final_results = {
        'Interaction (Base)': {k: {'hit': 0, 'ndcg': 0, 'mrr': 0} for k in K_VALUES},
        'Ultimate Re-rank (Dot Product)': {k: {'hit': 0, 'ndcg': 0, 'mrr': 0} for k in K_VALUES}
    }

    print(f"Evaluating Ultimate Two-Stage (Dot Product, Recall={RECALL_SIZE})...")
    
    with torch.no_grad():
        _, lightgcn_items = model.lightgcn.computer()
        padding = torch.zeros(1, lightgcn_items.shape[1], device=device)
        full_item_embeddings = torch.cat([padding, lightgcn_items], dim=0)

    for u in tqdm(test_users):
        gt = dataset.testDict[u]
        train = dataset.allPos[u]
        
        # --- Stage 1: Retrieval ---
        input_seq = [x + 1 for x in train[-MAXLEN:]]
        seq = np.zeros(MAXLEN, dtype=np.int32)
        seq[MAXLEN-len(input_seq):] = input_seq
        
        with torch.no_grad():
            log_feats = model.sasrec.log2feats(torch.LongTensor(seq).unsqueeze(0).to(device), item_emb_matrix=full_item_embeddings)
            user_query_rec = log_feats[:, -1, :] # (1, 64)
            
            i_scores = torch.matmul(user_query_rec, full_item_embeddings.t()).squeeze(0)[1:] 
            i_scores[train] = -1e9
            
            _, top_recall = torch.topk(i_scores, RECALL_SIZE)
            top_recall_np = top_recall.cpu().numpy()
            
        # --- Stage 2: Ultimate Re-ranking ---
        # 1. Base Retrieval Order
        p_base = top_recall_np
        
        # 2. Score Calculation & Z-Score Alignment
        # Calculate scores in Torch for Z-Score efficiency
        with torch.no_grad():
            # Use Decay-weighted average for User Query (Recent History prioritization)
            # Weights: [..., 0.8, 0.9, 1.0]
            num_hist = len(train)
            decay_weights = torch.linspace(0.5, 1.0, steps=num_hist).to(device).unsqueeze(1)
            
            u_s_weighted = (item_S[train] * decay_weights).sum(0, keepdim=True) / decay_weights.sum()
            u_c_weighted = (item_C[train] * decay_weights).sum(0, keepdim=True) / decay_weights.sum()
            
            # Dot Product scores for candidates
            s_scores = torch.matmul(u_s_weighted, item_S[top_recall].t()) # (1, RECALL)
            c_scores = torch.matmul(u_c_weighted, item_C[top_recall].t()) # (1, RECALL)
            i_scores_eval = i_scores[top_recall].unsqueeze(0) # (1, RECALL)
            
            # No normalization for I, Min-Max for S and C, then apply Multiplicative Boost
            i_raw = i_scores_eval.squeeze(0).cpu().numpy()
            s_norm = z_score_normalize(s_scores).squeeze(0).cpu().numpy()
            c_norm = z_score_normalize(c_scores).squeeze(0).cpu().numpy()
        
        # Weighting: Multiplicative Boost
        # We use I as the base, and let S & C act as small percentage boosts (+0% to +10% max)
        combined_score = i_raw * (1.0 + w_s * s_norm + w_c * c_norm)
        p_ultimate = top_recall_np[combined_score.argsort()[::-1]]
        
        # Collect Metrics
        for k in K_VALUES:
            h_b, n_b, m_b = compute_metrics(p_base, gt, k)
            final_results['Interaction (Base)'][k]['hit'] += h_b
            final_results['Interaction (Base)'][k]['ndcg'] += n_b
            final_results['Interaction (Base)'][k]['mrr'] += m_b
            
            h_u, n_u, m_u = compute_metrics(p_ultimate, gt, k)
            final_results['Ultimate Re-rank (Dot Product)'][k]['hit'] += h_u
            final_results['Ultimate Re-rank (Dot Product)'][k]['ndcg'] += n_u
            final_results['Ultimate Re-rank (Dot Product)'][k]['mrr'] += m_u
            
    num = len(test_users)
    print("\n" + "="*85)
    print(f"{'Method':<35} | {'K':<5} | {'Hit':<10} | {'NDCG':<10} | {'MRR':<10}")
    print("-" * 85)
    
    for m in ['Interaction (Base)', 'Ultimate Re-rank (Dot Product)']:
        for k in K_VALUES:
            res = final_results[m][k]
            print(f"{m:<35} | {k:<5} | {res['hit']/num:.4f}     | {res['ndcg']/num:.4f}     | {res['mrr']/num:.4f}")
    
    print("="*85)
    base_h20 = final_results['Interaction (Base)'][20]['hit'] / num
    ult_h20 = final_results['Ultimate Re-rank (Dot Product)'][20]['hit'] / num
    print(f"Ultimate Hit@20 Gain: {((ult_h20-base_h20)/base_h20*100):+.2f}%")
    print("="*85)

    # Output for ablation script to parse
    import json
    output_dict = {}
    for k in K_VALUES:
        output_dict[f'Hit@{k}'] = float(final_results['Ultimate Re-rank (Dot Product)'][k]['hit'] / num)
        output_dict[f'NDCG@{k}'] = float(final_results['Ultimate Re-rank (Dot Product)'][k]['ndcg'] / num)
        output_dict[f'MRR@{k}'] = float(final_results['Ultimate Re-rank (Dot Product)'][k]['mrr'] / num)
    print(f"ULTIMATE_FUSION_SCORE: {json.dumps(output_dict)}")

if __name__ == "__main__":
    main()

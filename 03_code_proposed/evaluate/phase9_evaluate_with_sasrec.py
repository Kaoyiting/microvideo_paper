#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 9: Evaluate with SASRec Sequence Prediction

Use SASRec's sequential modeling for recommendations instead of static embeddings.
This should reproduce the ~11% Hit@10 results.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
interaction_src = os.path.abspath(os.path.join(script_dir, "../../Interaction/src"))
if interaction_src not in sys.path:
    sys.path.insert(0, interaction_src)

from model import LightSASRec, SASRec
from dataloader import ShortVideoDataset

# Configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.abspath(os.path.join(script_dir, "../../Interaction/src/best_lightsasrec.pth"))
DATA_PATH = os.path.abspath(os.path.join(script_dir, "../../Interaction/data/shortvideo"))
ALIGNED_EMBEDDINGS_PATH = os.path.join(script_dir, "shortvideo_threepath_aligned_embeddings_1024dim.npz")
ALIGNMENT_MODEL_PATH = os.path.join(script_dir, "best_shortvideo_threepath_alignment.pth")

K_VALUES = [5, 10, 20]
MAXLEN = 50
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

class SASRecArgs:
    def __init__(self):
        self.hidden_units = 64
        self.num_blocks = 2
        self.num_heads = 1
        self.dropout_rate = 0.2
        self.maxlen = MAXLEN
        self.device = DEVICE
        self.norm_first = True

class ThreePathAlignmentModel(nn.Module):
    """Three-path alignment model"""
    def __init__(self, semantic_dim=1536, interaction_dim=64, content_dim=1024, output_dim=1024):
        super().__init__()
        
        self.semantic_proj = nn.Sequential(
            nn.Linear(semantic_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.interaction_proj = nn.Sequential(
            nn.Linear(interaction_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.content_proj = nn.Sequential(
            nn.Linear(content_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

def load_train_test_data():
    """Load train and test data"""
    train_dict = {}
    test_dict = {}
    
    with open(os.path.join(DATA_PATH, 'train.txt'), 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                user_id = int(parts[0])
                item_id = int(parts[1])
                if user_id not in train_dict:
                    train_dict[user_id] = []
                train_dict[user_id].append(item_id)
    
    with open(os.path.join(DATA_PATH, 'test.txt'), 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                user_id = int(parts[0])
                item_id = int(parts[1])
                if user_id not in test_dict:
                    test_dict[user_id] = []
                test_dict[user_id].append(item_id)
    
    return train_dict, test_dict

def compute_metrics(predictions, ground_truth, k_values):
    """Compute Hit, NDCG, MRR for given k values"""
    metrics = {f'{metric}@{k}': [] for metric in ['Hit', 'NDCG', 'MRR'] for k in k_values}
    
    for pred_items, true_items in zip(predictions, ground_truth):
        for k in k_values:
            pred_k = pred_items[:k]
            
            # Hit@K
            hit = 1.0 if len(set(pred_k) & set(true_items)) > 0 else 0.0
            metrics[f'Hit@{k}'].append(hit)
            
            # NDCG@K
            dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(pred_k) if item in true_items)
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(true_items), k)))
            ndcg = dcg / idcg if idcg > 0 else 0
            metrics[f'NDCG@{k}'].append(ndcg)
            
            # MRR@K
            for i, item in enumerate(pred_k):
                if item in true_items:
                    metrics[f'MRR@{k}'].append(1.0 / (i + 1))
                    break
            else:
                metrics[f'MRR@{k}'].append(0.0)
    
    return {k: np.mean(v) for k, v in metrics.items()}

def z_score_normalize(scores):
    """Apply Z-score normalization to scores"""
    mean = scores.mean(dim=1, keepdim=True)
    std = scores.std(dim=1, keepdim=True)
    return (scores - mean) / (std + 1e-8)

def evaluate_interaction_only(model, train_dict, test_dict, device):
    """Strategy: Interaction Only (Pure SASRec)"""
    print("\n[Strategy] Interaction Only (Pure SASRec)")
    
    model.eval()
    users = sorted(list(test_dict.keys()))
    users = [u for u in users if u in train_dict and len(train_dict[u]) > 0]
    
    all_predictions = []
    all_ground_truth = []
    
    batch_size = 100
    num_batches = (len(users) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for b in tqdm(range(num_batches), desc="Evaluating"):
            batch_users = users[b*batch_size : (b+1)*batch_size]
            
            seqs = np.zeros((len(batch_users), MAXLEN), dtype=np.int32)
            
            for i, uid in enumerate(batch_users):
                hist = train_dict[uid]
                input_seq = hist[-MAXLEN:]
                pad_len = MAXLEN - len(input_seq)
                if pad_len > 0:
                    seqs[i, pad_len:] = [x + 1 for x in input_seq]
                else:
                    seqs[i] = [x + 1 for x in input_seq]
            
            seqs_tensor = torch.LongTensor(seqs).to(device)
            
            # Get LightGCN item embeddings
            lightgcn_users, lightgcn_items = model.lightgcn.computer()
            padding = torch.zeros(1, lightgcn_items.shape[1], device=device)
            full_item_embeddings = torch.cat([padding, lightgcn_items], dim=0)
            
            # SASRec forward
            log_feats = model.sasrec.log2feats(seqs_tensor, full_item_embeddings)
            final_feat = log_feats[:, -1, :]
            
            # Combine LightGCN + SASRec
            static_feat = lightgcn_users[batch_users]
            combined_user_feat = (final_feat + static_feat) / 2.0
            
            # Score all items
            logits = combined_user_feat @ full_item_embeddings.t()
            logits[:, 0] = -float('inf')
            
            # Mask training items
            for i, uid in enumerate(batch_users):
                train_items = [x + 1 for x in train_dict[uid]]
                logits[i, train_items] = -float('inf')
            
            # Top K
            _, indices = torch.topk(logits, max(K_VALUES))
            indices = indices.cpu().numpy()
            
            for i, uid in enumerate(batch_users):
                pred_items = [x - 1 for x in indices[i]]
                all_predictions.append(pred_items)
                all_ground_truth.append(test_dict[uid])
    
    return compute_metrics(all_predictions, all_ground_truth, K_VALUES)

def evaluate_semantic_only(aligned_data, train_dict, test_dict, alignment_model, device):
    """Strategy: Semantic Only"""
    print("\n[Strategy] Semantic Only")
    
    semantic_embs = aligned_data['semantic']
    semantic_embs = torch.FloatTensor(semantic_embs).to(device)
    semantic_embs = F.normalize(semantic_embs, dim=1)
    
    users = sorted(list(test_dict.keys()))
    users = [u for u in users if u in train_dict and len(train_dict[u]) > 0]
    
    all_predictions = []
    all_ground_truth = []
    
    alignment_model.eval()
    
    with torch.no_grad():
        for uid in tqdm(users, desc="Evaluating"):
            # Compute user embedding as mean of interacted items (semantic space)
            train_items = train_dict[uid]
            user_emb = semantic_embs[train_items].mean(dim=0, keepdim=True)
            user_emb = F.normalize(user_emb, dim=1)
            
            # Compute similarities
            similarities = (user_emb @ semantic_embs.t()).squeeze()
            
            # Mask training items
            similarities[train_items] = -float('inf')
            
            # Top K
            _, indices = torch.topk(similarities, max(K_VALUES))
            pred_items = indices.cpu().numpy().tolist()
            
            all_predictions.append(pred_items)
            all_ground_truth.append(test_dict[uid])
    
    return compute_metrics(all_predictions, all_ground_truth, K_VALUES)

def evaluate_two_path(aligned_data, model, train_dict, test_dict, alignment_model, device, weights=(0.5, 0.5)):
    """Strategy: Two-Path (Semantic + Interaction)"""
    w_sem, w_int = weights
    print(f"\n[Strategy] Two-Path (Semantic {w_sem:.0%} + Interaction {w_int:.0%})")
    
    semantic_embs = torch.FloatTensor(aligned_data['semantic']).to(device)
    semantic_embs = F.normalize(semantic_embs, dim=1)
    
    model.eval()
    alignment_model.eval()
    
    users = sorted(list(test_dict.keys()))
    users = [u for u in users if u in train_dict and len(train_dict[u]) > 0]
    
    all_predictions = []
    all_ground_truth = []
    
    batch_size = 100
    num_batches = (len(users) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for b in tqdm(range(num_batches), desc="Evaluating"):
            batch_users = users[b*batch_size : (b+1)*batch_size]
            
            seqs = np.zeros((len(batch_users), MAXLEN), dtype=np.int32)
            
            for i, uid in enumerate(batch_users):
                hist = train_dict[uid]
                input_seq = hist[-MAXLEN:]
                pad_len = MAXLEN - len(input_seq)
                if pad_len > 0:
                    seqs[i, pad_len:] = [x + 1 for x in input_seq]
                else:
                    seqs[i] = [x + 1 for x in input_seq]
            
            seqs_tensor = torch.LongTensor(seqs).to(device)
            
            # SASRec scores
            lightgcn_users, lightgcn_items = model.lightgcn.computer()
            padding = torch.zeros(1, lightgcn_items.shape[1], device=device)
            full_item_embeddings = torch.cat([padding, lightgcn_items], dim=0)
            
            log_feats = model.sasrec.log2feats(seqs_tensor, full_item_embeddings)
            final_feat = log_feats[:, -1, :]
            
            # Combine LightGCN + SASRec
            static_feat = lightgcn_users[batch_users]
            combined_user_feat = (final_feat + static_feat) / 2.0
            
            # Project to 1024-dim
            combined_feat_1024 = alignment_model.interaction_proj(combined_user_feat)
            combined_feat_1024 = F.normalize(combined_feat_1024, dim=1)
            
            # Compute semantic scores
            semantic_scores_batch = []
            for i, uid in enumerate(batch_users):
                train_items = train_dict[uid]
                user_sem_emb = semantic_embs[train_items].mean(dim=0, keepdim=True)
                user_sem_emb = F.normalize(user_sem_emb, dim=1)
                sem_score = (user_sem_emb @ semantic_embs.t()).squeeze()
                semantic_scores_batch.append(sem_score)
            
            semantic_scores_batch = torch.stack(semantic_scores_batch)
            
            # Compute interaction scores (in 1024-dim aligned space)
            interaction_items_1024 = torch.FloatTensor(aligned_data['interaction']).to(device)
            interaction_items_1024 = F.normalize(interaction_items_1024, dim=1)
            
            interaction_scores = combined_feat_1024 @ interaction_items_1024.t()
            
            # Apply Z-score normalization to ensure balanced contribution
            semantic_scores_batch = z_score_normalize(semantic_scores_batch)
            interaction_scores = z_score_normalize(interaction_scores)
            
            # Fuse scores
            final_scores = w_sem * semantic_scores_batch + w_int * interaction_scores
            
            # Mask training items
            for i, uid in enumerate(batch_users):
                train_items = train_dict[uid]
                final_scores[i, train_items] = -float('inf')
            
            # Top K
            _, indices = torch.topk(final_scores, max(K_VALUES))
            indices = indices.cpu().numpy()
            
            for i, uid in enumerate(batch_users):
                pred_items = indices[i].tolist()
                all_predictions.append(pred_items)
                all_ground_truth.append(test_dict[uid])
    
    return compute_metrics(all_predictions, all_ground_truth, K_VALUES)

def evaluate_three_path(aligned_data, model, train_dict, test_dict, alignment_model, device, weights=(0.33, 0.33, 0.34)):
    """Strategy: Three-Path (Semantic + Interaction + Content)"""
    w_sem, w_int, w_con = weights
    print(f"\n[Strategy] Three-Path (I:{w_int:.0%} / S:{w_sem:.0%} / C:{w_con:.0%})")
    
    semantic_embs = torch.FloatTensor(aligned_data['semantic']).to(device)
    semantic_embs = F.normalize(semantic_embs, dim=1)
    
    content_embs = torch.FloatTensor(aligned_data['content']).to(device)
    content_embs = F.normalize(content_embs, dim=1)
    
    model.eval()
    alignment_model.eval()
    
    users = sorted(list(test_dict.keys()))
    users = [u for u in users if u in train_dict and len(train_dict[u]) > 0]
    
    all_predictions = []
    all_ground_truth = []
    
    batch_size = 100
    num_batches = (len(users) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for b in tqdm(range(num_batches), desc="Evaluating"):
            batch_users = users[b*batch_size : (b+1)*batch_size]
            
            seqs = np.zeros((len(batch_users), MAXLEN), dtype=np.int32)
            
            for i, uid in enumerate(batch_users):
                hist = train_dict[uid]
                input_seq = hist[-MAXLEN:]
                pad_len = MAXLEN - len(input_seq)
                if pad_len > 0:
                    seqs[i, pad_len:] = [x + 1 for x in input_seq]
                else:
                    seqs[i] = [x + 1 for x in input_seq]
            
            seqs_tensor = torch.LongTensor(seqs).to(device)
            
            # SASRec scores
            lightgcn_users, lightgcn_items = model.lightgcn.computer()
            padding = torch.zeros(1, lightgcn_items.shape[1], device=device)
            full_item_embeddings = torch.cat([padding, lightgcn_items], dim=0)
            
            log_feats = model.sasrec.log2feats(seqs_tensor, full_item_embeddings)
            final_feat = log_feats[:, -1, :]
            
            # Combine LightGCN static and SASRec sequential user embeddings
            # This implements the "LightGCN + SASRec" requirement
            static_feat = lightgcn_users[batch_users]
            combined_feat = (final_feat + static_feat) / 2.0
            
            # Project to 1024-dim for alignment with other paths
            combined_feat_1024 = alignment_model.interaction_proj(combined_feat)
            combined_feat_1024 = F.normalize(combined_feat_1024, dim=1)
            
            # Semantic scores
            semantic_scores_batch = []
            for i, uid in enumerate(batch_users):
                train_items = train_dict[uid]
                user_sem_emb = semantic_embs[train_items].mean(dim=0, keepdim=True)
                user_sem_emb = F.normalize(user_sem_emb, dim=1)
                sem_score = (user_sem_emb @ semantic_embs.t()).squeeze()
                semantic_scores_batch.append(sem_score)
            
            semantic_scores_batch = torch.stack(semantic_scores_batch)
            
            # Interaction scores (in aligned 1024-dim space)
            interaction_items_1024 = torch.FloatTensor(aligned_data['interaction']).to(device)
            interaction_items_1024 = F.normalize(interaction_items_1024, dim=1)
            interaction_scores = combined_feat_1024 @ interaction_items_1024.t()
            
            # Content scores
            content_scores_batch = []
            for i, uid in enumerate(batch_users):
                train_items = train_dict[uid]
                user_con_emb = content_embs[train_items].mean(dim=0, keepdim=True)
                user_con_emb = F.normalize(user_con_emb, dim=1)
                con_score = (user_con_emb @ content_embs.t()).squeeze()
                content_scores_batch.append(con_score)
            
            content_scores_batch = torch.stack(content_scores_batch)
            
            # Apply Z-score normalization for better path balancing
            semantic_scores_batch = z_score_normalize(semantic_scores_batch)
            interaction_scores = z_score_normalize(interaction_scores)
            content_scores_batch = z_score_normalize(content_scores_batch)
            
            # Fuse
            final_scores = (w_sem * semantic_scores_batch + 
                          w_int * interaction_scores + 
                          w_con * content_scores_batch)
            
            # Mask training items
            for i, uid in enumerate(batch_users):
                train_items = train_dict[uid]
                final_scores[i, train_items] = -float('inf')
            
            # Top K
            _, indices = torch.topk(final_scores, max(K_VALUES))
            indices = indices.cpu().numpy()
            
            for i, uid in enumerate(batch_users):
                pred_items = indices[i].tolist()
                all_predictions.append(pred_items)
                all_ground_truth.append(test_dict[uid])
    
    return compute_metrics(all_predictions, all_ground_truth, K_VALUES)

def print_results(results, title):
    """Print results in table format"""
    print(f"\n{'='*80}")
    print(title)
    print('='*80)
    print(f"\n{'Method':<30} {'Hit@5':>8} {'Hit@10':>8} {'Hit@20':>8} {'NDCG@5':>8} {'NDCG@10':>8} {'NDCG@20':>8} {'MRR@5':>8} {'MRR@10':>8} {'MRR@20':>8}")
    print('-'*80)
    
    for name, metrics in results.items():
        print(f"{name:<30} ", end='')
        for k in K_VALUES:
            print(f"{metrics[f'Hit@{k}']:>8.4f} ", end='')
        for k in K_VALUES:
            print(f"{metrics[f'NDCG@{k}']:>8.4f} ", end='')
        for k in K_VALUES:
            print(f"{metrics[f'MRR@{k}']:>8.4f} ", end='')
        print()

def main():
    print("="*80)
    print("Phase 9: Evaluate with SASRec Sequence Prediction")
    print("="*80)
    
    # Load data
    print("\n[1/5] 載入資料...")
    train_dict, test_dict = load_train_test_data()
    print(f"        訓練用戶: {len(train_dict)}")
    print(f"        測試用戶: {len(test_dict)}")
    
    print("  載入對齊 embeddings...")
    aligned_data = np.load(ALIGNED_EMBEDDINGS_PATH)
    print(f"        Semantic: {aligned_data['semantic'].shape}")
    print(f"        Interaction: {aligned_data['interaction'].shape}")
    print(f"        Content: {aligned_data['content'].shape}")
    
    # Load dataset (for model initialization)
    dataset = ShortVideoDataset(path=DATA_PATH)
    
    # Load model
    print("\n[2/5] 載入模型...")
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"        設備: {device}")
    
    config = {
        'latent_dim_rec': 64,
        'lightGCN_n_layers': 3,
        'dropout': 0,
        'keep_prob': 0.6,
    }
    
    sasrec_args = SASRecArgs()
    model = LightSASRec(config, sasrec_args, dataset)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'))
    model = model.to(device)
    model.eval()
    
    # Load alignment model
    print("  載入對齊模型...")
    alignment_model = ThreePathAlignmentModel()
    alignment_checkpoint = torch.load(ALIGNMENT_MODEL_PATH, map_location='cpu')
    alignment_model.load_state_dict(alignment_checkpoint['model_state_dict'])
    alignment_model = alignment_model.to(device)
    alignment_model.eval()
    
    # Evaluate
    print("\n[3/5] 評估不同策略...")
    results = {}
    
    # Interaction Only
    results['Interaction Only'] = evaluate_interaction_only(model, train_dict, test_dict, device)
    
    # Semantic Only
    results['Semantic Only'] = evaluate_semantic_only(aligned_data, train_dict, test_dict, alignment_model, device)
    
    # Two-Path Average
    results['Two-Path Avg'] = evaluate_two_path(aligned_data, model, train_dict, test_dict, alignment_model, device, weights=(0.5, 0.5))
    
    # Two-Path Weighted
    results['Two-Path Weighted'] = evaluate_two_path(aligned_data, model, train_dict, test_dict, alignment_model, device, weights=(0.4, 0.6))
    
    # Three-Path Average
    results['Three-Path Avg'] = evaluate_three_path(aligned_data, model, train_dict, test_dict, alignment_model, device, weights=(0.33, 0.33, 0.34))
    
    # Three-Path Weighted variants
    results['Weighted 1 (60/20/20)'] = evaluate_three_path(aligned_data, model, train_dict, test_dict, alignment_model, device, weights=(0.20, 0.60, 0.20))
    
    results['Weighted 2 (80/10/10)'] = evaluate_three_path(aligned_data, model, train_dict, test_dict, alignment_model, device, weights=(0.10, 0.80, 0.10))
    
    results['Weighted 3 (70/15/15)'] = evaluate_three_path(aligned_data, model, train_dict, test_dict, alignment_model, device, weights=(0.15, 0.70, 0.15))
    
    # Print results
    print("\n[4/5] 結果總結...")
    print_results(results, "Content - ShortVideo 評估結果")
    
    print("\n" + "="*80)
    print("Phase 9 完成！")
    print("="*80)

if __name__ == "__main__":
    main()

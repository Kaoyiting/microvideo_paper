#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 11: Evaluate Fusion MLP Reranking (ShortVideo)

Pipeline:
1. Retrieval: Use Z-Score weighted dot product to get Top-100 candidates.
2. Reranking: Use trained Fusion MLP to score Top-100 candidates.
3. Compute metrics: Hit@K, NDCG@K, MRR@K.
"""

import os
import sys
import numpy as np
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

# ============ Configuration ============
CHECKPOINT_PATH = os.path.abspath(os.path.join(script_dir, "../../Interaction/src/best_lightsasrec.pth"))
DATA_PATH = os.path.abspath(os.path.join(script_dir, "../../Interaction/data/shortvideo"))
ALIGNED_EMBEDDINGS_PATH = os.path.join(script_dir, "shortvideo_threepath_aligned_embeddings_1024dim.npz")
ALIGNMENT_MODEL_PATH = os.path.join(script_dir, "best_shortvideo_threepath_alignment.pth")
FUSION_MODEL_PATH = os.path.join(script_dir, "best_fusion_mlp.pth")

K_VALUES = [5, 10, 20]
RETRIEVAL_TOP_K = 100  # Number of candidates to rerank
MAXLEN = 50
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ============ Model Definitions ============

class ThreePathAlignmentModel(nn.Module):
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

class GatedFusionMLP(nn.Module):
    """
    Gated MLP to fuse user and item features from three paths.
    Uses a User-Specific Gate to weight the importance of S/I/C paths for the item.
    """
    def __init__(self, embedding_dim=1024): 
        super().__init__()
        
        # User Gate: Takes User's S, I, C features and outputs weights for Item's S, I, C
        self.user_gate = nn.Sequential(
            nn.Linear(embedding_dim * 3, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 3), # [w_s, w_i, w_c]
            nn.Softmax(dim=1)
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 6, 2048),
            nn.BatchNorm1d(2048),
            nn.SiLU(),
            nn.Dropout(0.3),
            
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 256),
            nn.SiLU(),
            
            nn.Linear(256, 1) # Raw Score
        )

    def forward(self, user_vecs, item_vecs):
        # user_vecs: [B, 3072] -> [S, I, C]
        # item_vecs: [B, 3072] -> [S, I, C]
        
        # Compute Gate Weights [B, 3]
        gate_weights = self.user_gate(user_vecs) 
        w_s, w_i, w_c = gate_weights[:, 0:1], gate_weights[:, 1:2], gate_weights[:, 2:3]
        
        # Apply weights to ITEM features
        i_s, i_i, i_c = item_vecs.chunk(3, dim=1)
        weighted_item = torch.cat([
            i_s * (1 + w_s), 
            i_i * (1 + w_i), 
            i_c * (1 + w_c)
        ], dim=1)
        
        # Concatenate User + Weighted Item
        x = torch.cat([user_vecs, weighted_item], dim=1)
        return self.mlp(x)

class SASRecArgs:
    def __init__(self):
        self.hidden_units = 64
        self.num_blocks = 2
        self.num_heads = 1
        self.dropout_rate = 0.2
        self.maxlen = MAXLEN
        self.device = DEVICE
        self.norm_first = True

# ============ Utilities ============

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

def print_results(results, title):
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

# ============ Evaluation Logic ============

def evaluate_with_reranking(aligned_data, sasrec_model, align_model, fusion_model, train_dict, test_dict, device):
    """
    Evaluate using 2-stage pipeline: Retrieval (Z-Score Weighted) -> Reranking (Fusion MLP)
    Retrieval weights: 0.5/0.1/0.4 (Interaction/Semantic/Content) - optimized from Phase 9
    """
    print("\n[Evaluation] Fusion MLP Reranking...")
    
    # [1] Prepare Item Embeddings
    sem_items = torch.FloatTensor(aligned_data['semantic']).to(device)
    int_items = torch.FloatTensor(aligned_data['interaction']).to(device)
    con_items = torch.FloatTensor(aligned_data['content']).to(device)
    
    sem_items = F.normalize(sem_items, dim=1)
    int_items = F.normalize(int_items, dim=1)
    con_items = F.normalize(con_items, dim=1)
    
    # Concatenated Item Features for MLP [N, 3072]
    item_feats_concat = torch.cat([sem_items, int_items, con_items], dim=1)
    
    # [2] Prepare Models
    sasrec_model.eval()
    align_model.eval()
    fusion_model.eval()
    
    # LightGCN + SASRec Base
    lightgcn_users, lightgcn_items = sasrec_model.lightgcn.computer()
    padding = torch.zeros(1, lightgcn_items.shape[1], device=device)
    full_item_embeddings = torch.cat([padding, lightgcn_items], dim=0)
    
    users = sorted(list(test_dict.keys()))
    # Filter users with training history
    users = [u for u in users if u in train_dict and len(train_dict[u]) > 0]
    
    all_predictions = []
    all_ground_truth = []
    
    batch_size = 100 # Batch of users
    num_batches = (len(users) + batch_size - 1) // batch_size
    
    # Weights for Retrieval (Use pure interaction for Step 1 to ensure recall)
    w_int, w_sem, w_con = 1.0, 0.0, 0.0 
    
    with torch.no_grad():
        for b in tqdm(range(num_batches), desc="Evaluating"):
            batch_users = users[b*batch_size : (b+1)*batch_size]
            
            # --- FEATURE GENERATION ---
            
            # 1. User Interaction Features (SASRec)
            seqs = np.zeros((len(batch_users), MAXLEN), dtype=np.int32)
            for i, uid in enumerate(batch_users):
                hist = train_dict[uid][-MAXLEN:]
                pad_len = MAXLEN - len(hist)
                if pad_len > 0:
                    seqs[i, pad_len:] = [x + 1 for x in hist]
                else:
                    seqs[i] = [x + 1 for x in hist]
            
            seqs_tensor = torch.LongTensor(seqs).to(device)
            log_feats = sasrec_model.sasrec.log2feats(seqs_tensor, full_item_embeddings)
            final_feat = log_feats[:, -1, :] 
            static_feat = lightgcn_users[batch_users]
            combined_feat = (final_feat + static_feat) / 2.0
            
            # Project to 1024
            user_int_1024 = align_model.interaction_proj(combined_feat)
            user_int_1024 = F.normalize(user_int_1024, dim=1)
            
            # 2. User Semantic & Content Features (Mean Pooling)
            batch_sem_list = []
            batch_con_list = []
            
            for uid in batch_users:
                items = train_dict[uid]
                u_sem = sem_items[items].mean(dim=0)
                u_sem = F.normalize(u_sem, dim=0)
                batch_sem_list.append(u_sem)
                
                u_con = con_items[items].mean(dim=0)
                u_con = F.normalize(u_con, dim=0)
                batch_con_list.append(u_con)
                
            user_sem_1024 = torch.stack(batch_sem_list)
            user_con_1024 = torch.stack(batch_con_list)
            
            # User Features Concatenated [B, 3072]
            user_feats_concat = torch.cat([user_sem_1024, user_int_1024, user_con_1024], dim=1)
            
            # --- STAGE 1: RETRIEVAL (Top-100) ---
            
            # Calculate Scores
            s_sem = user_sem_1024 @ sem_items.t()
            s_int = user_int_1024 @ int_items.t()
            s_con = user_con_1024 @ con_items.t()
            
            # Z-Score Normalize
            s_sem = z_score_normalize(s_sem)
            s_int = z_score_normalize(s_int)
            s_con = z_score_normalize(s_con)
            
            # Weighted Sum
            retrieval_scores = w_int * s_int + w_sem * s_sem + w_con * s_con
            
            # Mask training items
            for i, uid in enumerate(batch_users):
                train_items = train_dict[uid]
                retrieval_scores[i, train_items] = -float('inf')
                
            # Get Top-K Candidates from Retrieval
            _, top_k_indices = torch.topk(retrieval_scores, RETRIEVAL_TOP_K) # [B, 100]
            
            # --- STAGE 2: RERANKING (Residaul Fusion MLP) ---
            
            batch_preds = []
            
            for i, uid in enumerate(batch_users):
                # Get candidate items
                candidates = top_k_indices[i] 
                
                # 1. Get MLP Scores (Based on Aligned Features)
                candidate_feats = item_feats_concat[candidates]
                user_feat_expanded = user_feats_concat[i].unsqueeze(0).expand(RETRIEVAL_TOP_K, -1)
                
                # Apply Sigmoid because current GatedFusionMLP outputs raw scores
                mlp_scores = torch.sigmoid(fusion_model(user_feat_expanded, candidate_feats).squeeze()) 
                
                # 2. Get Original SASRec Scores (Strong Baseline)
                # We need to re-compute precise scores for these 100 candidates using original embeddings
                # combined_feat[i]: [64]
                # lightgcn_items: [N, 64]
                candidate_raw_embs = lightgcn_items[candidates] # [100, 64]
                sasrec_raw_scores = candidate_raw_embs @ combined_feat[i] # [100] dot product
                
                # Normalize SASRec scores to be comparable with MLP Sigmoid output
                # Simple Min-Max scaling per user session
                sasrec_min = sasrec_raw_scores.min()
                sasrec_max = sasrec_raw_scores.max()
                if sasrec_max > sasrec_min:
                    sasrec_norm = (sasrec_raw_scores - sasrec_min) / (sasrec_max - sasrec_min)
                else:
                    sasrec_norm = torch.zeros_like(sasrec_raw_scores)
                
                # 3. Residual Fusion: Base + Delta
                # Base: Original SASRec Signal (Strong)
                # Delta: MLP Prediction (Correction from Semantic/Content)
                # Formula: Score = 0.7 * SASRec + 0.3 * MLP
                final_scores = 0.7 * sasrec_norm + 0.3 * mlp_scores
                
                # Sort candidates by Final Score
                _, rerank_indices = torch.sort(final_scores, descending=True)
                
                # Map back to original item IDs
                final_item_ids = candidates[rerank_indices].cpu().numpy()
                batch_preds.append(final_item_ids.tolist())
                all_ground_truth.append(test_dict[uid])
                
            all_predictions.extend(batch_preds)
            
    return compute_metrics(all_predictions, all_ground_truth, K_VALUES)
    
def main():
    print("="*80)
    print("Phase 11: Evaluate Fusion MLP Reranking")
    print("="*80)
    
    # [1] Load Data
    aligned_data = np.load(ALIGNED_EMBEDDINGS_PATH)
    
    train_dict = {}
    test_dict = {}
    
    with open(os.path.join(DATA_PATH, 'train.txt'), 'r') as f:
        for line in f:
            u, i = map(int, line.split())
            if u not in train_dict: train_dict[u] = []
            train_dict[u].append(i)
            
    with open(os.path.join(DATA_PATH, 'test.txt'), 'r') as f:
        for line in f:
            u, i = map(int, line.split())
            if u not in test_dict: test_dict[u] = []
            test_dict[u].append(i)
            
    # [2] Load Models
    device = torch.device(DEVICE)
    dataset = ShortVideoDataset(path=DATA_PATH)
    config = {'latent_dim_rec': 64, 'lightGCN_n_layers': 3, 'dropout': 0, 'keep_prob': 0.6}
    sasrec_args = SASRecArgs()
    
    sasrec_model = LightSASRec(config, sasrec_args, dataset)
    sasrec_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'))
    sasrec_model = sasrec_model.to(device)
    
    align_model = ThreePathAlignmentModel()
    align_model.load_state_dict(torch.load(ALIGNMENT_MODEL_PATH, map_location='cpu')['model_state_dict'])
    align_model = align_model.to(device)
    
    fusion_model = GatedFusionMLP()
    fusion_model.load_state_dict(torch.load(FUSION_MODEL_PATH, map_location='cpu'))
    fusion_model = fusion_model.to(device)
    
    # [3] Evaluate
    results = {}
    results['Fusion MLP Rerank'] = evaluate_with_reranking(
        aligned_data, sasrec_model, align_model, fusion_model, train_dict, test_dict, device
    )
    
    # [4] Print Results
    print_results(results, "Content - ShortVideo Fusion MLP Results")

if __name__ == "__main__":
    main()

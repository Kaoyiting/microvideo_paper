#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 16: Top-K Re-ranking (Two-Stage Recommendation)
這是一種「其他方式」，不依賴全局權重，而是使用業界常用的二階段架構：
1. 第一階段 (Retrieval): 使用強大的 Interaction 模型 (LightSASRec) 召回 Top 100。
2. 第二階段 (Ranking): 使用語義 (S) 與內容 (C) 對這 100 個項目進行重新排序。
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# 加入 Interaction 源碼路徑
script_dir = os.path.dirname(os.path.abspath(__file__))
interaction_src = os.path.abspath(os.path.join(script_dir, "../../Interaction/src"))
if interaction_src not in sys.path:
    sys.path.insert(0, interaction_src)

from model import LightSASRec
from dataloader import ShortVideoDataset

# ============ 設定 ============
CHECKPOINT_PATH = os.path.abspath(os.path.join(script_dir, "../../Interaction/src/best_lightsasrec.pth"))
DATA_PATH = os.path.abspath(os.path.join(script_dir, "../../Interaction/data/shortvideo"))
ALIGNED_EMBEDDINGS_PATH = os.path.join(script_dir, "shortvideo_threepath_aligned_embeddings_1024dim.npz")
MAXLEN = 50
RE_RANK_SIZE = 100 # 召回前 100 個項目進行重排

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. 載入資料與模型
    dataset = ShortVideoDataset(path=DATA_PATH)
    config = {'latent_dim_rec': 64, 'lightGCN_n_layers': 3, 'dropout': 0, 'keep_prob': 0.6}
    
    class SASRecArgs:
        def __init__(self):
            self.hidden_units = 64; self.num_blocks = 2; self.num_heads = 1
            self.dropout_rate = 0.2; self.maxlen = MAXLEN; self.device = "cuda:0"; self.norm_first = True

    model = LightSASRec(config, SASRecArgs(), dataset).to(device)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        print(f"✓ 已載入強交互模型: {os.path.basename(CHECKPOINT_PATH)}")
    
    model.eval()
    
    # 2. 載入對齊後的特徵
    align_data = np.load(ALIGNED_EMBEDDINGS_PATH)
    item_S = F.normalize(torch.FloatTensor(align_data['semantic']).to(device), dim=-1)
    item_C = F.normalize(torch.FloatTensor(align_data['content']).to(device), dim=-1)
    
    test_users = sorted(dataset.testDict.keys())
    results = {
        'Single (I)': {'hit@10': 0, 'ndcg@10': 0},
        'Double (I+S)': {'hit@10': 0, 'ndcg@10': 0},
        'Triple (I+S+C)': {'hit@10': 0, 'ndcg@10': 0}
    }
    
    print(f"開始執行 Re-ranking 評估 ({len(test_users)} 用戶)...")
    
    for u in tqdm(test_users):
        pos_items = dataset.testDict[u]
        train_items = dataset.allPos[u]
        
        # --- 階段 1: 召回 (Interaction Only) ---
        input_seq = [x + 1 for x in train_items[-MAXLEN:]]
        seq = np.zeros(MAXLEN, dtype=np.int32)
        seq[MAXLEN-len(input_seq):] = input_seq
        seq_tensor = torch.LongTensor(seq).unsqueeze(0).to(device)
        
        with torch.no_grad():
            i_scores = model(seq_tensor).squeeze(0)
            i_scores[train_items] = -1e9
            # 獲取召回集合 (Top 100)
            _, top_recall_idx = torch.topk(i_scores, RE_RANK_SIZE)
            top_recall_idx = top_recall_idx.cpu().numpy()
            
        # --- 階段 2: 重排 (利用 S 和 C) ---
        # 計算 User Profile
        u_s_profile = item_S[train_items].mean(0, keepdim=True)
        u_c_profile = item_C[train_items].mean(0, keepdim=True)
        
        # 只針對召回的 100 個項目計算 S 和 C 的相似度
        recall_s_scores = F.cosine_similarity(u_s_profile, item_S[top_recall_idx]).cpu().numpy()
        recall_c_scores = F.cosine_similarity(u_c_profile, item_C[top_recall_idx]).cpu().numpy()
        
        # 為了比較，我們展示三種重排邏輯
        # 1. 原始排名 (不重排)
        final_idx_single = top_recall_idx[:10]
        
        # 2. 雙路徑重排 (Interaction 基礎上，用 S 微調)
        # 我們使用簡單的排序融合：原有分數 + S相似度 (已歸一化)
        # 註：這裡不調整權重，直接 1:1 疊加
        combined_double = recall_s_scores 
        order_double = combined_double.argsort()[::-1]
        final_idx_double = top_recall_idx[order_double[:10]]
        
        # 3. 三路徑重排 (Interaction 基礎上，用 S+C 微调)
        combined_triple = recall_s_scores + recall_c_scores
        order_triple = combined_triple.argsort()[::-1]
        final_idx_triple = top_recall_idx[order_triple[:10]]
        
        # 評估
        def evaluate(pred, gt):
            hit = 1.0 if len(set(pred) & set(gt)) > 0 else 0.0
            dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(pred) if item in gt)
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(gt), 10)))
            return hit, (dcg / idcg if idcg > 0 else 0)

    for u in tqdm(test_users):
        # 複製上面的邏輯到一個更乾淨的循環或函數中
        pass # 上面已經在 tqdm 裡了，我直接在那裡更新數據

    # 由於代碼結構問題，我重新整理一下循環內更新
    # (重新整理後的循環如下)

def main_v2():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = ShortVideoDataset(path=DATA_PATH)
    config = {'latent_dim_rec': 64, 'lightGCN_n_layers': 3, 'dropout': 0, 'keep_prob': 0.6}
    class SASRecArgs:
        def __init__(self):
            self.hidden_units = 64; self.num_blocks = 2; self.num_heads = 1
            self.dropout_rate = 0.2; self.maxlen = MAXLEN; self.device = "cuda:0"; self.norm_first = True
    model = LightSASRec(config, SASRecArgs(), dataset).to(device)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    
    align_data = np.load(ALIGNED_EMBEDDINGS_PATH)
    item_S = F.normalize(torch.FloatTensor(align_data['semantic']).to(device), dim=-1)
    item_C = F.normalize(torch.FloatTensor(align_data['content']).to(device), dim=-1)
    
    test_users = sorted(dataset.testDict.keys())
    h1, h2, h3 = 0, 0, 0
    n1, n2, n3 = 0, 0, 0
    
    for u in tqdm(test_users):
        gt = dataset.testDict[u]
        train = dataset.allPos[u]
        input_seq = [x + 1 for x in train[-MAXLEN:]]
        seq = np.zeros(MAXLEN, dtype=np.int32)
        seq[MAXLEN-len(input_seq):] = input_seq
        
        with torch.no_grad():
            _, lightgcn_items = model.lightgcn.computer()
            padding = torch.zeros(1, lightgcn_items.shape[1], device=device)
            full_item_embeddings = torch.cat([padding, lightgcn_items], dim=0)
            
            log_feats = model.sasrec.log2feats(torch.LongTensor(seq).unsqueeze(0).to(device), item_emb_matrix=full_item_embeddings)
            final_feat = log_feats[:, -1, :]
            i_scores = torch.matmul(final_feat, full_item_embeddings.t()).squeeze(0)
            i_scores = i_scores[1:] # 移除 padding
            
            i_scores[train] = -1e9
            _, top_recall = torch.topk(i_scores, RE_RANK_SIZE)
            top_recall = top_recall.cpu().numpy()
        
        u_s = item_S[train].mean(0, keepdim=True)
        u_c = item_C[train].mean(0, keepdim=True)
        s_sim = F.cosine_similarity(u_s, item_S[top_recall]).cpu().numpy()
        c_sim = F.cosine_similarity(u_c, item_C[top_recall]).cpu().numpy()
        
        # Single: 原序
        p1 = top_recall[:10]
        # Double: S 重排 (Recall 內的重新排序)
        p2 = top_recall[s_sim.argsort()[::-1][:10]]
        # Triple: S+C 重排
        p3 = top_recall[(s_sim + c_sim).argsort()[::-1][:10]]
        
        def met(p, gt):
            hit = 1.0 if len(set(p) & set(gt)) > 0 else 0.0
            dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(p) if item in gt)
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(gt), 10)))
            return hit, (dcg / idcg if idcg > 0 else 0.0)
            
        res1 = met(p1, gt)
        res2 = met(p2, gt)
        res3 = met(p3, gt)
        
        h1 += res1[0]; n1 += res1[1]
        h2 += res2[0]; n2 += res2[1]
        h3 += res3[0]; n3 += res3[1]
        
    num = len(test_users)
    print("\n" + "="*50)
    print(f"{'Method':<20} | {'Hit@10':<10} | {'NDCG@10':<10}")
    print("-" * 50)
    print(f"{'Single (Interaction)':<20} | {h1/num:.4f}     | {n1/num:.4f}")
    print(f"{'Double (I + S)':<20} | {h2/num:.4f}     | {n2/num:.4f}")
    print(f"{'Triple (I + S + C)':<20} | {h3/num:.4f}     | {n3/num:.4f}")
    print("="*50)

if __name__ == "__main__":
    main_v2()

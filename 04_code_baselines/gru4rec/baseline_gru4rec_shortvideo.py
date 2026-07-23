#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline: GRU4Rec (ShortVideo)
Objective: Sequential recommendation using RNN/GRU.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '../../Interaction/src')))
sys.path.append(os.path.abspath(os.path.join(script_dir, '../../content')))

from dataloader import ShortVideoDataset # GRU4Rec script previously used MicroLensDataset which might be outdated
from unified_evaluator import RecommendationEvaluator, print_results

# Hyperparameters
BATCH_SIZE = 256
LR = 0.001
EMB_DIM = 64
HIDDEN_DIM = 128
EPOCHS = 30
MAX_LEN = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GRU4Rec(nn.Module):
    def __init__(self, n_items, emb_dim, hidden_dim):
        super(GRU4Rec, self).__init__()
        self.item_emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_items + 1)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, seq):
        x = self.item_emb(seq) 
        _, h = self.gru(x) 
        out = self.fc(h.squeeze(0)) 
        return out

class SeqDataset(Dataset):
    def __init__(self, user_train, n_items, max_len):
        self.samples = []
        for u in user_train:
            seq = user_train[u]
            if len(seq) < 2: continue
            for i in range(1, len(seq)):
                in_seq = seq[max(0, i-max_len):i]
                target = seq[i]
                pad_len = max_len - len(in_seq)
                in_seq = [0] * pad_len + in_seq
                self.samples.append((in_seq, target))
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        seq, target = self.samples[idx]
        return torch.LongTensor(seq), target

def train_gru4rec():
    print("="*80)
    print("GRU4Rec Baseline for ShortVideo")
    print("="*80)
    
    # Load ShortVideo data
    dataset = ShortVideoDataset(path=os.path.abspath(os.path.join(script_dir, "../../Interaction/data/shortvideo")))
    n_items = dataset.m_item

    # Prepare sequences for training/evaluation
    user_train = {}
    for u, i in zip(dataset.trainUser, dataset.trainItem):
        if u not in user_train: user_train[u] = []
        user_train[u].append(i + 1)
        
    user_test_seq = {}
    for u in user_train:
        seq = user_train[u]
        pad_len = MAX_LEN - len(seq)
        if pad_len > 0:
            user_test_seq[u] = [0] * pad_len + seq
        else:
            user_test_seq[u] = seq[-MAX_LEN:]

    train_loader = DataLoader(SeqDataset(user_train, n_items, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
    
    model = GRU4Rec(n_items, EMB_DIM, HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for seq, target in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            seq, target = seq.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            logits = model(seq)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
    # 1. Prepare evaluation data
    model.eval()
    print("\nEvaluating GRU4Rec...")
    
    # 建立評估專用的用戶列表，排除沒有訓練序列或測試資料的
    eval_u_indices = []
    user_test_items = []
    user_train_mask = []
    
        # 使用 dataset.testDict 獲取測試樣本
    test_dict = dataset.testDict
    sorted_users = sorted(test_dict.keys())
    
    for u in sorted_users:
        if u not in user_train or len(user_train[u]) == 0:
            continue
        
        # 正樣本 (0-indexed)
        test_items = [i for i in test_dict[u]]
        if len(test_items) > 0:
            eval_u_indices.append(u)
            user_test_items.append(test_items)
            
            # 使用於排除已訓練過的物品 (0-indexed)
            train_items = [i-1 for i in user_train[u]]
            user_train_mask.append(train_items)

    # 2. Get User Features (Hidden States) and Item Features (Embedding Matrix)
    user_feats_list = []
    with torch.no_grad():
        for i in range(0, len(eval_u_indices), BATCH_SIZE):
            batch_u = eval_u_indices[i:i+BATCH_SIZE]
            batch_seqs = []
            for u in batch_u:
                hist = user_train[u][-MAX_LEN:]
                pad_len = MAX_LEN - len(hist)
                padded = [0] * pad_len + hist
                batch_seqs.append(padded)
            
            batch_seq_torch = torch.LongTensor(batch_seqs).to(DEVICE)
            # 獲取 GRU 的隱狀態作為用戶特徵
            # model.gru 返回 (output, h_n), 其中 h_n 是最後一個隱狀態
            x = model.item_emb(batch_seq_torch)
            _, h = model.gru(x)
            user_feats = h.squeeze(0) # [batch, hidden]
            user_feats_list.append(user_feats)
            
    all_user_feats = torch.cat(user_feats_list, dim=0)
    
    # 獲取物品的所有嵌入 (排除 padding idx 0)
    # model.fc 將 hidden 映射到 n_items+1, 我們可以直接用其投影後的空間或原始 embedding
    # 為了與 Unified Evaluator 的相似度計算一致，我們直接獲取所有物品的 logits 投影矩陣
    # 或者，最簡單的方式是直接跑過 model 的 fc 層
    item_embs = model.fc.weight[1:, :] # [n_items, hidden]
    
    # 3. 使用 Unified Evaluator
    evaluator = RecommendationEvaluator(device=DEVICE.type)
    results = evaluator.evaluate(
        all_user_feats, 
        item_embs, 
        user_test_items, 
        k_list=[5, 10, 20], 
        train_items=user_train_mask
    )
    print_results(results, "GRU4Rec (ShortVideo)")

if __name__ == "__main__":
    train_gru4rec()

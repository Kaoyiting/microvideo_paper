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

from dataloader import ShortVideoDataset # ShortVideo uses this
from unified_evaluator import RecommendationEvaluator, print_results

# Hyperparameters
BATCH_SIZE = 512
LR = 0.001
EMB_DIM = 64
HIDDEN_DIM = 128
EPOCHS = 30
MAX_LEN = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GRU4RecModel(nn.Module):
    def __init__(self, n_items, emb_dim, hidden_dim):
        super(GRU4RecModel, self).__init__()
        self.item_emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_items + 1)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, seq):
        x = self.item_emb(seq)
        _, h = self.gru(x)
        # Return h for unified evaluator, and initial predictions if needed
        return h[-1] # Shape: [batch_size, hidden_dim]

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
    
    base_dir = r"c:\Users\kao\microvideo"
    data_dir = os.path.join(base_dir, "Interaction", "data", "shortvideo")
    dataset = ShortVideoDataset(path=data_dir)
    n_items = dataset.m_item

    # --- Reconstruct user interaction dictionary ---
    user_train_dict = {}
    train_users = set()
    # Assuming dataset.trainUser and dataset.trainItem arrays exist
    # If not, use dataset.allPos minus last item?
    # ShortVideoDataset usually has trainUser/trainItem similar to LightGCN
    user_train_dict = dataset.allPos # Use allPos but exclude last for training? No, usually split train/test
    # Check dataset structure carefully
    
    # Actually, dataset.trainUser and dataset.testUser partition the interactions?
    # Based on LightGCN usage, yes.
    # Reconstruct user sequences:
    user_train_seq = {}
    for i in range(len(dataset.trainUser)):
        u = dataset.trainUser[i]
        item = dataset.trainItem[i]
        if u not in user_train_seq: user_train_seq[u] = []
        user_train_seq[u].append(item + 1) # 1-based indexing for padding 0

    user_test_targets = {} # Ground Truth for Test
    for u in dataset.testDict:
        user_test_targets[u] = [i + 1 for i in dataset.testDict[u]]

    train_loader = DataLoader(SeqDataset(user_train_seq, n_items, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
    
    model = GRU4RecModel(n_items, EMB_DIM, HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for seq, target in pbar:
            seq, target = seq.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            h = model(seq) # [batch, hidden]
            logits = model.fc(h) # [batch, n_items+1]
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
    model.eval()
    print("\nEvaluating GRU4Rec...")
    
    eval_u_indices = []
    eval_u_reps = [] # User representations (hidden states)
    user_input_seqs = []

    # Filter users: only eval users in test set
    test_users = sorted(user_test_targets.keys())
    
    for u in test_users:
        if u in user_train_seq:
            eval_u_indices.append(u)
            seq = user_train_seq[u]
            pad_len = MAX_LEN - len(seq)
            if pad_len > 0:
                user_input_seqs.append([0] * pad_len + seq)
            else:
                user_input_seqs.append(seq[-MAX_LEN:])
    
    # Batch process to get user representations
    with torch.no_grad():
        for i in range(0, len(user_input_seqs), BATCH_SIZE):
            batch_seq = torch.LongTensor(user_input_seqs[i:i+BATCH_SIZE]).to(DEVICE)
            h = model(batch_seq) 
            eval_u_reps.append(h)
            
    if len(eval_u_reps) > 0:
        user_embeddings = torch.cat(eval_u_reps, dim=0) # [N_test_users, hidden_dim]
        item_embeddings = model.fc.weight # [N_items+1, hidden_dim]
        
        # We need to act carefully: item 0 is padding. Evaluator handles 0-based indices or what?
        # ShortVideoDataset uses 0-based items, but GRU4Rec uses 1-based (0 is padding).
        # UnifiedEvaluator usually expects 0-based item indices OR mapped ones. 
        # The user_test_targets are now 1-based (because we did item+1).
        # item_embeddings[0] corresponds to padding.
        # item_embeddings[1] corresponds to item 0 in original ID.
        # So evaluating dot product of user_emb with item_emb[1:] will align with 0-based targets?
        # No, unified evaluator does dot product of USER x ITEM.T.
        # If we pass item_embeddings[1:], then index 0 corresponds to item 1 (original 0).
        # And our ground truth targets should be 0-based if we pass item_embeddings[1:].
        
        # Let's align to 0-based for UnifiedEvaluator:
        # 1. User Reps: OK
        # 2. Item Reps: item_embeddings[1:] (shape: [N_items, hidden_dim])
        # 3. Ground Truth: dataset.testDict (0-based)
        # 4. Train Items (for masking): dataset.allPos (0-based)
        
        ground_truth_list = [dataset.testDict[u] for u in eval_u_indices]
        train_items_list = [dataset.allPos[u] for u in eval_u_indices]
        
        evaluator = RecommendationEvaluator(device=DEVICE)
        metrics = evaluator.evaluate(
            user_embeddings, 
            item_embeddings[1:], 
            ground_truth_list, 
            k_list=[5, 10, 20],
            train_items=train_items_list
        )
        print_results(metrics, "GRU4Rec (ShortVideo)")
    else:
        print("No users to evaluate!")

if __name__ == "__main__":
    train_gru4rec()

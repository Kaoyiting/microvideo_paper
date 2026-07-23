#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 8: Train Three-Path Alignment Model (ShortVideo)
- Semantic embeddings: 1536-dim (OpenAI)
- Interaction embeddings: 64-dim (LightGCN)
- Content embeddings: 1024-dim (ImageBind Vision+Audio)
- Align all three to 1024-dim space using contrastive learning
"""

import os
import json
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ============ Configuration ============
# 注意：所有路徑已仔細確認

# Input paths
SEMANTIC_EMBEDDING_PATH = r"E:\semantic_shortvideo_with_embeddings.csv"
INTERACTION_EMBEDDING_PATH = r"../../Interaction/src/best_lightsasrec.pth"
CONTENT_EMBEDDING_PATH = "shortvideo_content_embeddings_1024dim.npz"
DATA_INDEX_PATH = "data_index_shortvideo.json"
TRAIN_DATA_PATH = r"../../Interaction/data/shortvideo/train.txt"
TEST_DATA_PATH = r"../../Interaction/data/shortvideo/test.txt"

# Output paths
OUTPUT_MODEL_PATH = "best_shortvideo_threepath_alignment.pth"
OUTPUT_ALIGNED_EMBEDDINGS_PATH = "shortvideo_threepath_aligned_embeddings_1024dim.npz"

# Hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
TEMPERATURE = 0.03  # More strict for better alignment precision
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ============ Model Definition ============

class ThreePathAlignmentModel(nn.Module):
    """
    Three-path alignment model with projection layers
    """
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
        
        # Content path: 1024 -> 1024 (identity or fine-tuning)
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

# ============ Dataset ============

class ThreePathDataset(Dataset):
    """
    Dataset for three-path alignment
    """
    def __init__(self, semantic_emb, interaction_emb, content_emb, item_ids):
        self.semantic_emb = semantic_emb
        self.interaction_emb = interaction_emb
        self.content_emb = content_emb
        self.item_ids = item_ids
        
    def __len__(self):
        return len(self.item_ids)
    
    def __getitem__(self, idx):
        item_id = self.item_ids[idx]
        
        return {
            'semantic': torch.FloatTensor(self.semantic_emb[item_id]),
            'interaction': torch.FloatTensor(self.interaction_emb[item_id]),
            'content': torch.FloatTensor(self.content_emb[item_id]),
            'item_id': item_id
        }

# ============ Loss Functions ============

def info_nce_loss(emb1, emb2, temperature=0.15):
    """
    InfoNCE contrastive loss between two embedding sets
    """
    # emb1, emb2: (batch_size, dim), already normalized
    
    # Compute similarity matrix: (batch_size, batch_size)
    logits = torch.matmul(emb1, emb2.T) / temperature
    
    # Labels: diagonal elements are positive pairs
    labels = torch.arange(logits.shape[0], device=logits.device)
    
    # Cross-entropy loss
    loss = F.cross_entropy(logits, labels)
    
    return loss

def three_path_contrastive_loss(outputs, temperature=0.15):
    """
    Compute weighted contrastive loss for all three pairs:
    - Semantic vs Interaction (0.2)
    - Semantic vs Content (0.4)
    - Interaction vs Content (0.4)
    Lower weight on Interaction-only path to force model to learn better Semantic/Content representations.
    """
    semantic = outputs['semantic']
    interaction = outputs['interaction']
    content = outputs['content']
    
    # Three pairwise losses
    loss_sem_int = info_nce_loss(semantic, interaction, temperature)
    loss_sem_con = info_nce_loss(semantic, content, temperature)
    loss_int_con = info_nce_loss(interaction, content, temperature)
    
    # Weighted loss: Emphasis on Content and Semantic
    total_loss = (0.2 * loss_sem_int + 0.4 * loss_sem_con + 0.4 * loss_int_con)
    
    return total_loss, {
        'sem_int': loss_sem_int.item(),
        'sem_con': loss_sem_con.item(),
        'int_con': loss_int_con.item()
    }

# ============ Data Loading ============

def load_data():
    """
    Load all three types of embeddings
    Returns: semantic_array, interaction_array, content_array, valid_mapped_ids, num_total_items
    """
    print("\n[1/7] 載入資料...")
    
    # [1] Load data index to get ID mappings
    print("  [1.1] 讀取資料索引...")
    with open(DATA_INDEX_PATH, 'r', encoding='utf-8') as f:
        data_index = json.load(f)
    
    # Get valid items (both have semantic and interaction)
    valid_items = {int(k): v for k, v in data_index.items() if v.get('valid', False)}
    print(f"        ✓ 有效項目數: {len(valid_items)}")
    
    # [2] Load semantic embeddings (1536-dim)
    print("  [1.2] 載入語義 embeddings...")
    df_semantic = pd.read_csv(SEMANTIC_EMBEDDING_PATH)
    
    # Create semantic dict: original_item_id -> embedding
    semantic_dict = {}
    for _, row in df_semantic.iterrows():
        original_id = str(row['original_item_id'])
        # Parse string representation of array
        embedding_str = row['embedding']
        if isinstance(embedding_str, str):
            embedding = np.array(ast.literal_eval(embedding_str), dtype=np.float32)
        else:
            embedding = np.array(embedding_str, dtype=np.float32)
        semantic_dict[original_id] = embedding
    print(f"        ✓ 載入 {len(semantic_dict)} 個語義 embeddings (1536 維)")
    
    # [3] Load interaction embeddings (64-dim from LightGCN)
    print("  [1.3] 載入交互 embeddings...")
    checkpoint = torch.load(INTERACTION_EMBEDDING_PATH, map_location='cpu')
    
    # Extract LightGCN item embeddings from checkpoint
    # Format: checkpoint['lightgcn.embedding_item.weight'] with shape [5907, 64]
    if 'lightgcn.embedding_item.weight' in checkpoint:
        interaction_emb = checkpoint['lightgcn.embedding_item.weight'].numpy()
    elif 'model' in checkpoint and 'embedding_item.weight' in checkpoint['model']:
        interaction_emb = checkpoint['model']['embedding_item.weight'].numpy()
    elif 'embedding_item.weight' in checkpoint:
        interaction_emb = checkpoint['embedding_item.weight'].numpy()
    else:
        raise KeyError(f"Cannot find item embeddings in checkpoint. Keys: {checkpoint.keys()}")
    
    print(f"        ✓ 載入交互 embeddings，形狀: {interaction_emb.shape}")
    print(f"        ✓ 使用前 {len(data_index)} 個 (對應 mapped_item_id 0-{len(data_index)-1})")
    
    # Only use first N rows corresponding to items (mapped_id 0 to N-1)
    interaction_emb = interaction_emb[:len(data_index)]
    
    # [4] Load content embeddings (1024-dim)
    print("  [1.4] 載入內容 embeddings...")
    content_data = np.load(CONTENT_EMBEDDING_PATH)
    content_embeddings = content_data['embeddings']  # (N, 1024)
    content_mapped_ids = content_data['mapped_ids']  # (N,)
    
    # Create content dict: mapped_id -> embedding
    content_dict = {}
    for mapped_id, emb in zip(content_mapped_ids, content_embeddings):
        content_dict[int(mapped_id)] = emb
    print(f"        ✓ 載入 {len(content_dict)} 個內容 embeddings (1024 維)")
    
    # [5] Create aligned arrays by mapped_id
    print("  [1.5] 對齊三路 embeddings...")
    
    # Initialize arrays (size = total items in data_index)
    num_total_items = len(data_index)
    semantic_array = np.zeros((num_total_items, 1536), dtype=np.float32)
    interaction_array = np.zeros((num_total_items, 64), dtype=np.float32)
    content_array = np.zeros((num_total_items, 1024), dtype=np.float32)
    
    valid_mapped_ids = []
    
    for mapped_id, item_info in valid_items.items():
        original_id = item_info['original_item_id']
        
        # Check if all three embeddings exist
        has_semantic = original_id in semantic_dict
        has_interaction = mapped_id < len(interaction_emb)
        has_content = mapped_id in content_dict
        
        if has_semantic and has_interaction and has_content:
            semantic_array[mapped_id] = semantic_dict[original_id]
            interaction_array[mapped_id] = interaction_emb[mapped_id]
            content_array[mapped_id] = content_dict[mapped_id]
            valid_mapped_ids.append(mapped_id)
    
    print(f"        ✓ 三路對齊完成: {len(valid_mapped_ids)} 個項目")
    print(f"        ✓ 覆蓋率: {len(valid_mapped_ids)/len(valid_items)*100:.2f}%")
    
    return semantic_array, interaction_array, content_array, valid_mapped_ids, num_total_items

# ============ Training ============

def train_model():
    """
    Train three-path alignment model
    """
    print("="*80)
    print("Phase 8: Train Three-Path Alignment Model (ShortVideo)")
    print("="*80)
    
    # Load data
    semantic_array, interaction_array, content_array, valid_mapped_ids, num_total_items = load_data()
    
    # Create dataset and dataloader
    print("\n[2/7] 建立 DataLoader...")
    
    # Convert to dict format for dataset
    semantic_dict = {mid: semantic_array[mid] for mid in valid_mapped_ids}
    interaction_dict = {mid: interaction_array[mid] for mid in valid_mapped_ids}
    content_dict = {mid: content_array[mid] for mid in valid_mapped_ids}
    
    dataset = ThreePathDataset(semantic_dict, interaction_dict, content_dict, valid_mapped_ids)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    print(f"     ✓ 資料集項目數: {len(dataset)}")
    print(f"     ✓ Batch size: {BATCH_SIZE}")
    print(f"     ✓ 每個 epoch 批次數: {len(dataloader)}")
    
    # Create model
    print("\n[3/7] 建立模型...")
    model = ThreePathAlignmentModel(
        semantic_dim=1536,
        interaction_dim=64,
        content_dim=1024,
        output_dim=1024
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"     ✓ 模型參數: {total_params:,} 總參數, {trainable_params:,} 可訓練")
    print(f"     ✓ 設備: {DEVICE}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print(f"\n[4/7] 開始訓練...")
    print(f"     Epochs: {NUM_EPOCHS}")
    print(f"     Learning Rate: {LEARNING_RATE}")
    print(f"     Temperature: {TEMPERATURE}")
    print()
    
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        epoch_losses = {'sem_int': 0, 'sem_con': 0, 'int_con': 0}
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", ncols=100)
        
        for batch in pbar:
            semantic = batch['semantic'].to(DEVICE)
            interaction = batch['interaction'].to(DEVICE)
            content = batch['content'].to(DEVICE)
            
            # Forward pass
            outputs = model(semantic=semantic, interaction=interaction, content=content)
            
            # Compute loss
            loss, loss_dict = three_path_contrastive_loss(outputs, TEMPERATURE)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            for k, v in loss_dict.items():
                epoch_losses[k] += v
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'si': f"{loss_dict['sem_int']:.3f}",
                'sc': f"{loss_dict['sem_con']:.3f}",
                'ic': f"{loss_dict['int_con']:.3f}"
            })
        
        # Epoch summary
        avg_loss = epoch_loss / len(dataloader)
        avg_losses = {k: v / len(dataloader) for k, v in epoch_losses.items()}
        
        print(f"  Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | "
              f"SI: {avg_losses['sem_int']:.3f} | "
              f"SC: {avg_losses['sem_con']:.3f} | "
              f"IC: {avg_losses['int_con']:.3f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, OUTPUT_MODEL_PATH)
    
    print(f"\n[5/7] 訓練完成！最佳 loss: {best_loss:.4f}")
    print(f"     ✓ 模型已儲存: {OUTPUT_MODEL_PATH}")
    
    # Generate aligned embeddings
    print(f"\n[6/7] 生成對齊後的 embeddings...")
    model.eval()
    
    aligned_semantic = np.zeros((num_total_items, 1024), dtype=np.float32)
    aligned_interaction = np.zeros((num_total_items, 1024), dtype=np.float32)
    aligned_content = np.zeros((num_total_items, 1024), dtype=np.float32)
    
    with torch.no_grad():
        for mapped_id in tqdm(valid_mapped_ids, desc="Generating", ncols=100):
            semantic = torch.FloatTensor(semantic_array[mapped_id:mapped_id+1]).to(DEVICE)
            interaction = torch.FloatTensor(interaction_array[mapped_id:mapped_id+1]).to(DEVICE)
            content = torch.FloatTensor(content_array[mapped_id:mapped_id+1]).to(DEVICE)
            
            outputs = model(semantic=semantic, interaction=interaction, content=content)
            
            aligned_semantic[mapped_id] = outputs['semantic'].cpu().numpy()
            aligned_interaction[mapped_id] = outputs['interaction'].cpu().numpy()
            aligned_content[mapped_id] = outputs['content'].cpu().numpy()
    
    # Save aligned embeddings
    np.savez(
        OUTPUT_ALIGNED_EMBEDDINGS_PATH,
        semantic=aligned_semantic,
        interaction=aligned_interaction,
        content=aligned_content,
        valid_ids=np.array(valid_mapped_ids)
    )
    
    print(f"     ✓ 對齊 embeddings 已儲存: {OUTPUT_ALIGNED_EMBEDDINGS_PATH}")
    
    # Statistics
    print(f"\n[7/7] 統計資訊...")
    print(f"\n{'='*80}")
    print("Phase 8 完成！")
    print(f"{'='*80}")
    print(f"訓練項目數:        {len(valid_mapped_ids)}")
    print(f"Embedding 維度:    1024")
    print(f"最佳 Loss:         {best_loss:.6f}")
    print(f"\n輸出檔案:")
    print(f"  ✓ 模型: {OUTPUT_MODEL_PATH}")
    print(f"    大小: {os.path.getsize(OUTPUT_MODEL_PATH) / 1024**2:.2f} MB")
    print(f"  ✓ 對齊 embeddings: {OUTPUT_ALIGNED_EMBEDDINGS_PATH}")
    print(f"    大小: {os.path.getsize(OUTPUT_ALIGNED_EMBEDDINGS_PATH) / 1024**2:.2f} MB")
    print(f"{'='*80}")
    print(f"\n下一步: Phase 9 - 評估推薦效果")
    print("="*80)

# ============ Main ============

if __name__ == "__main__":
    train_model()

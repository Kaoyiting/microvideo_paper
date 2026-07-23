#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 7: Generate ImageBind Content Embeddings for ShortVideo
- Load preprocessed frames (5 per video) and audio (2s per video)
- Generate vision + audio embeddings using ImageBind
- Fusion: (vision + audio) / 2 to get 1024-dim embedding per video
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Add ImageBind to path
IMAGEBIND_PATH = os.path.abspath("../../external/ImageBind")
if IMAGEBIND_PATH not in sys.path:
    sys.path.insert(0, IMAGEBIND_PATH)

from imagebind import data as imagebind_data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

# ============ Configuration ============
DATA_INDEX_PATH = "data_index_shortvideo.json"
FRAMES_DIR = r"E:\short-video-dataset\frames_5samples"
AUDIO_DIR = r"E:\short-video-dataset\audio_2s"
OUTPUT_EMBEDDING_PATH = "shortvideo_content_embeddings_1024dim.npz"

BATCH_SIZE = 16  # Process 16 videos at a time
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
NUM_FRAMES = 5

# ============ Helper Functions ============

def load_frames_for_item(item_id, num_frames=5):
    """
    Load 5 frames for a single video
    Returns: List of PIL Images or None
    
    Note: Phase 6 saves frames as {original_id}_frame_0.jpg, {original_id}_frame_1.jpg, etc.
    """
    # Phase 6 格式：{original_id}_frame_0.jpg
    frame_paths = [os.path.join(FRAMES_DIR, f"{item_id}_frame_{i}.jpg") for i in range(num_frames)]
    
    images = []
    for frame_path in frame_paths:
        if not os.path.exists(frame_path):
            return None
        try:
            img = Image.open(frame_path).convert("RGB")
            images.append(img)
        except:
            return None
    
    return images

def transform_images(images, device):
    """
    Transform images using ImageBind's preprocessing
    Returns: Tensor of shape (num_frames, 3, 224, 224)
    """
    from torchvision import transforms
    
    data_transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])
    
    transformed = []
    for img in images:
        transformed.append(data_transform(img))
    
    return torch.stack(transformed, dim=0).to(device)

def load_audio_for_item(item_id):
    """
    Load audio path for a single video
    Returns: audio path or None
    """
    audio_path = os.path.join(AUDIO_DIR, f"{item_id}.wav")
    if os.path.exists(audio_path):
        return audio_path
    return None

def generate_embeddings(model, data_index, device):
    """
    Generate vision + audio fusion embeddings for all videos
    Returns: 
        - embeddings_array: (N, 1024) numpy array
        - mapped_ids: list of mapped_item_id (0-5906)
    """
    embeddings_list = []
    mapped_ids_list = []
    
    # Get valid items (only those with valid=True)
    valid_items = [(k, v) for k, v in data_index.items() if v.get('valid', False)]
    
    print(f"Processing {len(valid_items)} valid videos...")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Device: {device}")
    print()
    
    success_count = 0
    fail_count = 0
    
    with torch.no_grad():
        for i in tqdm(range(0, len(valid_items), BATCH_SIZE), desc="Generating embeddings", ncols=100):
            batch_items = valid_items[i:i+BATCH_SIZE]
            
            batch_mapped_ids = []
            batch_frames = []
            batch_audio_paths = []
            
            # Load data for this batch
            for mapped_id, item_info in batch_items:
                original_id = item_info['original_item_id']
                
                # Load frames
                images = load_frames_for_item(original_id, NUM_FRAMES)
                if images is None:
                    fail_count += 1
                    continue
                
                # Load audio path
                audio_path = load_audio_for_item(original_id)
                
                # Transform images
                transformed = transform_images(images, device)  # (5, 3, 224, 224)
                batch_frames.append(transformed)
                batch_audio_paths.append(audio_path)
                batch_mapped_ids.append(int(mapped_id))
            
            if len(batch_frames) == 0:
                continue
            
            # ===== Vision Encoding =====
            # Stack all frames: (batch_size * 5, 3, 224, 224)
            all_frames = torch.cat(batch_frames, dim=0)
            
            inputs = {ModalityType.VISION: all_frames}
            embeddings = model(inputs)
            vision_embeddings = embeddings[ModalityType.VISION]  # (batch_size * 5, 1024)
            
            # Reshape and max pool: (batch_size, 5, 1024) -> (batch_size, 1024)
            vision_embeddings = vision_embeddings.view(len(batch_mapped_ids), NUM_FRAMES, 1024)
            pooled_vision = torch.max(vision_embeddings, dim=1)[0]  # (batch_size, 1024)
            
            # ===== Audio Encoding =====
            # Load audio data using ImageBind's data loader
            valid_audio_indices = []
            audio_data_list = []
            
            for idx, audio_path in enumerate(batch_audio_paths):
                if audio_path is not None:
                    try:
                        audio_data = imagebind_data.load_and_transform_audio_data([audio_path], device)
                        audio_data_list.append(audio_data[0])  # (128, 204)
                        valid_audio_indices.append(idx)
                    except:
                        pass
            
            # ===== Fusion =====
            if len(audio_data_list) > 0:
                # Encode audio
                audio_tensor = torch.stack(audio_data_list, dim=0)  # (num_audio, 128, 204)
                audio_inputs = {ModalityType.AUDIO: audio_tensor}
                audio_embeddings = model(audio_inputs)
                audio_emb = audio_embeddings[ModalityType.AUDIO]  # (num_audio, 1024)
                
                # Fusion: average of vision and audio
                for j, idx in enumerate(valid_audio_indices):
                    fused = (pooled_vision[idx] + audio_emb[j]) / 2.0
                    embeddings_list.append(fused.cpu().numpy())
                    mapped_ids_list.append(batch_mapped_ids[idx])
                    success_count += 1
                
                # For items without audio, use vision only
                for idx in range(len(batch_mapped_ids)):
                    if idx not in valid_audio_indices:
                        embeddings_list.append(pooled_vision[idx].cpu().numpy())
                        mapped_ids_list.append(batch_mapped_ids[idx])
                        success_count += 1
            else:
                # No audio in this batch, use vision only
                for idx in range(len(batch_mapped_ids)):
                    embeddings_list.append(pooled_vision[idx].cpu().numpy())
                    mapped_ids_list.append(batch_mapped_ids[idx])
                    success_count += 1
    
    print(f"\n✓ 成功生成 {success_count} 個 embeddings")
    print(f"✗ 失敗 {fail_count} 個項目")
    
    # Convert to numpy array and sort by mapped_id
    embeddings_array = np.array(embeddings_list)  # (N, 1024)
    mapped_ids_array = np.array(mapped_ids_list)
    
    # Sort by mapped_id
    sort_indices = np.argsort(mapped_ids_array)
    embeddings_array = embeddings_array[sort_indices]
    mapped_ids_array = mapped_ids_array[sort_indices]
    
    return embeddings_array, mapped_ids_array

# ============ Main ============

def main():
    print("="*80)
    print("Phase 7: Generate ImageBind Content Embeddings (ShortVideo)")
    print("="*80)
    
    # [1/6] Load data index
    print(f"\n[1/6] 讀取資料索引...")
    with open(DATA_INDEX_PATH, 'r', encoding='utf-8') as f:
        data_index = json.load(f)
    
    valid_count = sum(1 for v in data_index.values() if v.get('valid', False))
    print(f"   總項目數: {len(data_index)}")
    print(f"   有效項目數: {valid_count}")
    
    # [2/6] Setup device
    print(f"\n[2/6] 設置計算設備...")
    device = DEVICE
    print(f"   使用設備: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   顯存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("   ⚠ 未檢測到 GPU，使用 CPU（速度較慢）")
    
    # [3/6] Load ImageBind model
    print(f"\n[3/6] 載入 ImageBind 模型...")
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)
    print("   ✓ 模型載入完成")
    
    # [4/6] Generate embeddings
    print(f"\n[4/6] 生成 Content Embeddings...")
    print("   - Vision: 5 幀 → Max Pooling → 1024 維")
    print("   - Audio: 2 秒 → Encoding → 1024 維")
    print("   - Fusion: (Vision + Audio) / 2 → 1024 維")
    print()
    
    embeddings_array, mapped_ids = generate_embeddings(model, data_index, device)
    
    # [5/6] Save embeddings
    print(f"\n[5/6] 儲存 embeddings...")
    np.savez(
        OUTPUT_EMBEDDING_PATH,
        embeddings=embeddings_array,
        mapped_ids=mapped_ids
    )
    print(f"   ✓ 已儲存至: {OUTPUT_EMBEDDING_PATH}")
    
    # [6/6] Statistics
    print(f"\n[6/6] 統計資訊...")
    print(f"\n{'='*80}")
    print("Phase 7 完成！")
    print(f"{'='*80}")
    print(f"處理項目數:        {len(embeddings_array)}")
    print(f"Embedding 維度:    {embeddings_array.shape[1]}")
    print(f"Embedding 類型:    {embeddings_array.dtype}")
    print(f"平均值:            {embeddings_array.mean():.6f}")
    print(f"標準差:            {embeddings_array.std():.6f}")
    print(f"最小值:            {embeddings_array.min():.6f}")
    print(f"最大值:            {embeddings_array.max():.6f}")
    print(f"\n✓ 輸出檔案: {OUTPUT_EMBEDDING_PATH}")
    print(f"  檔案大小: {os.path.getsize(OUTPUT_EMBEDDING_PATH) / 1024**2:.2f} MB")
    print(f"{'='*80}")
    print(f"\n下一步: Phase 8 - 訓練三路對齊模型")
    print("="*80)

if __name__ == "__main__":
    main()

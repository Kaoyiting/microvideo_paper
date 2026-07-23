import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import sys
import torch
import numpy as np
import json

# Add paths to use existing models and dataloaders
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(ARTIFACT_ROOT, ".."))
SHARED_SRC = os.path.join(ARTIFACT_ROOT, "04_code_baselines", "shared_src")
for path in [SHARED_SRC, os.path.join(REPO_ROOT, "Interaction", "src"), os.path.join(REPO_ROOT, "content")]:
    if path not in sys.path:
        sys.path.insert(0, path)
from dataloader import ShortVideoDataset
from model import LightGCN
from unified_evaluator import RecommendationEvaluator, print_results

def solve():
    print("="*80)
    print("LightGCN Baseline for ShortVideo")
    print("="*80)
    
    data_dir = os.path.join(ARTIFACT_ROOT, "01_data_processed", "shortvideo")
    ckpt_path = os.path.join(ARTIFACT_ROOT, "06_checkpoints_selected", "best_shortvideo_official.pth")
    aligned_embeddings_path = os.path.join(
        ARTIFACT_ROOT,
        "02_features_embeddings",
        "aligned_threepath_embeddings",
        "shortvideo_threepath_aligned_embeddings_1024dim.npz",
    )
    id_mapping_path = os.path.join(data_dir, "id_mappings.json")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Dataset
    dataset = ShortVideoDataset(path=data_dir)
    
    # 2. Extract Test Items
    test_items_dict = {}
    for user, item in zip(dataset.testUser, dataset.testItem):
        if user not in test_items_dict: test_items_dict[user] = []
        test_items_dict[user].append(item)
    
    # Extract Train Items for masking
    train_items_dict = {}
    for user, item in zip(dataset.trainUser, dataset.trainItem):
        if user not in train_items_dict: train_items_dict[user] = []
        train_items_dict[user].append(item)
    
    # 3. Load Aligned Item Search Space
    data_npz = np.load(aligned_embeddings_path, allow_pickle=True)
    # Note: ShortVideo uses 'valid_ids' instead of 'item_ids'
    aligned_item_ids = [str(tid) for tid in data_npz['valid_ids']]
    
    if os.path.exists(id_mapping_path):
        with open(id_mapping_path, 'r') as f:
            id_mappings = json.load(f)
        idx_to_item_id = {int(v): str(k) for k, v in id_mappings['item_to_idx'].items()}
    else:
        # Fallback if mapping not found, assume index is id
        idx_to_item_id = {i: str(i) for i in range(dataset.m_item)}
        
    item_id_to_aligned_idx = {tid: idx for idx, tid in enumerate(aligned_item_ids)}
    
    # 4. Initialize Model
    config = {
        'latent_dim_rec': 64,
        'lightGCN_n_layers': 3,
        'keep_prob': 1.0,
        'dropout': False
    }
    model = LightGCN(config, dataset)
    
    # 5. Load Weights
    if os.path.exists(ckpt_path):
        print(f"Loading weights from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        model_dict = model.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            # Check for lightgcn prefix in the checkpoint keys
            key = k.replace('lightgcn.', '')
            if key in model_dict and v.shape == model_dict[key].shape:
                new_state_dict[key] = v
            # Also try if the model expects the prefix but the checkpoint doesn't have it
            elif f'lightgcn.{k}' in model_dict and v.shape == model_dict[f'lightgcn.{k}'].shape:
                new_state_dict[f'lightgcn.{k}'] = v
            elif k in model_dict and v.shape == model_dict[k].shape:
                new_state_dict[k] = v
        
        print(f"Successfully loaded {len(new_state_dict)}/{len(model_dict)} parameters.")
        model.load_state_dict(new_state_dict, strict=False)
    else:
        print(f"Warning: No checkpoint found at {ckpt_path}")

    model = model.to(device)
    model.eval()

    # 6. Evaluate
    print("Evaluating...")
    with torch.no_grad():
        # Get all embeddings via LightGCN propagation
        all_user_emb, all_item_emb = model.computer()
        
        # Map aligned items
        if 'id_mappings' in locals() and id_mappings is not None:
            item_to_idx_ds = id_mappings['item_to_idx']
            ordered_ds_indices = []
            for tid in aligned_item_ids:
                if tid in item_to_idx_ds:
                    ordered_ds_indices.append(item_to_idx_ds[tid])
                else:
                    ordered_ds_indices.append(0)
        else:
            # ShortVideo fallback: aligned_item_ids are the indices themselves
            ordered_ds_indices = [int(tid) for tid in aligned_item_ids]
        
        ordered_ds_indices = torch.LongTensor(ordered_ds_indices).to(device)
        item_embs_aligned = all_item_emb[ordered_ds_indices]
        
        # Prepare test data
        sorted_users = sorted(list(test_items_dict.keys()))
        user_test_items = []
        user_train_items = []
        eval_u_indices = []
        for user in sorted_users:
            test_ids = [str(idx_to_item_id[idx]) for idx in test_items_dict[user] if idx in idx_to_item_id]
            test_filtered = [item_id_to_aligned_idx[tid] for tid in test_ids if tid in item_id_to_aligned_idx]
            if len(test_filtered) > 0:
                user_test_items.append(test_filtered)
                eval_u_indices.append(user)

                # Extract and map train items for masking
                if user in train_items_dict:
                    t_ids = [str(idx_to_item_id[idx]) for idx in train_items_dict[user] if idx in idx_to_item_id]
                    t_filtered = [item_id_to_aligned_idx[tid] for tid in t_ids if tid in item_id_to_aligned_idx]
                    user_train_items.append(t_filtered)
                else:
                    user_train_items.append([])
        
        user_indices_torch = torch.LongTensor(eval_u_indices).to(device)
        user_embs_eval = all_user_emb[user_indices_torch]
        
        evaluator = RecommendationEvaluator(device=device)
        metrics = evaluator.evaluate(user_embs_eval, item_embs_aligned, user_test_items, k_list=[5, 10, 20], train_items=user_train_items)
        
    print_results(metrics, "LightGCN (ShortVideo)")

if __name__ == "__main__":
    solve()

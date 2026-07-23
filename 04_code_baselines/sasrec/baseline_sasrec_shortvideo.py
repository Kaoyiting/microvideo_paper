import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import sys
import torch
import numpy as np
import json
from tqdm import tqdm

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(ARTIFACT_ROOT, ".."))
SHARED_SRC = os.path.join(ARTIFACT_ROOT, "04_code_baselines", "shared_src")
for path in [SHARED_SRC, os.path.join(REPO_ROOT, "Interaction", "src"), os.path.join(REPO_ROOT, "content")]:
    if path not in sys.path:
        sys.path.insert(0, path)
from dataloader import ShortVideoDataset
from model import LightSASRec
from unified_evaluator import RecommendationEvaluator, print_results

class SASRecArgs:
    def __init__(self, device):
        self.hidden_units = 64
        self.num_blocks = 2
        self.num_heads = 1
        self.dropout_rate = 0.2
        self.maxlen = 50
        self.device = device
        self.norm_first = False

def solve():
    print("="*80)
    print("SASRec Baseline for ShortVideo")
    print("="*80)
    
    data_dir = os.path.join(ARTIFACT_ROOT, "01_data_processed", "shortvideo")
    ckpt_path = os.path.join(ARTIFACT_ROOT, "06_checkpoints_selected", "best_lightsasrec.pth")
    aligned_embeddings_path = os.path.join(
        ARTIFACT_ROOT,
        "02_features_embeddings",
        "aligned_threepath_embeddings",
        "shortvideo_threepath_aligned_embeddings_1024dim.npz",
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Dataset
    dataset = ShortVideoDataset(path=data_dir)
    
    # 2. Load Aligned Item Search Space
    data_npz = np.load(aligned_embeddings_path, allow_pickle=True)
    aligned_item_ids = [str(tid) for tid in data_npz['valid_ids']]
    item_id_to_aligned_idx = {tid: idx for idx, tid in enumerate(aligned_item_ids)}
    
    # 3. Extract Test Items & Train Sequences
    test_dict = {}
    with open(os.path.join(data_dir, 'test.txt')) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                u, i = int(parts[0]), int(parts[1])
                if u not in test_dict: test_dict[u] = []
                test_dict[u].append(i)
                
    train_seqs = {}
    with open(os.path.join(data_dir, 'train.txt')) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                u, i = int(parts[0]), int(parts[1])
                if u not in train_seqs: train_seqs[u] = []
                train_seqs[u].append(i)

    # 3. Initialize Model
    config = {
        'latent_dim_rec': 64,
        'lightGCN_n_layers': 3,
        'keep_prob': 1.0,
        'dropout': False,
        'maxlen': 50,
        'sasrec_num_blocks': 2,
        'sasrec_num_heads': 4,
        'sasrec_dropout': 0.5,
        'sasrec_norm_first': False,
        'lr': 0.001,
        'decay': 0.0001,
        'epochs': 1
    }
    sas_args = SASRecArgs(device)
    model = LightSASRec(config, sas_args, dataset)
    
    # 4. Load Weights
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

    model = model.to(device)
    model.eval()

    # 5. Evaluate
    print("Evaluating...")
    maxlen = config['maxlen']
    eval_u_indices = []
    user_test_items = []
    user_train_items = []
    
    sorted_users = sorted(test_dict.keys())
    for user in sorted_users:
        if user not in train_seqs or len(train_seqs[user]) == 0: continue
        
        # Map test items and filter those not in aligned search space
        test_ids = [str(i) for i in test_dict[user]]
        test_filtered = [item_id_to_aligned_idx[tid] for tid in test_ids if tid in item_id_to_aligned_idx]
        
        if len(test_filtered) > 0:
            eval_u_indices.append(user)
            user_test_items.append(test_filtered)
            
            # Map train items for masking
            train_ids = [str(i) for i in train_seqs[user]]
            train_filtered = [item_id_to_aligned_idx[tid] for tid in train_ids if tid in item_id_to_aligned_idx]
            user_train_items.append(train_filtered)

    with torch.no_grad():
        _, lightgcn_items = model.lightgcn.computer()
        
        # Prepare aligned item embeddings (0-indexed to match item_id_to_aligned_idx)
        ordered_ds_indices = [int(tid) for tid in aligned_item_ids]
        ordered_ds_indices = torch.LongTensor(ordered_ds_indices).to(device)
        item_embs_aligned = lightgcn_items[ordered_ds_indices]
        
        # For SASRec seq encoding, we still need the original full item embeddings (padded)
        padding = torch.zeros(1, lightgcn_items.shape[1], device=device)
        full_item_embs_for_seq = torch.cat([padding, lightgcn_items], dim=0)
        
        batch_size = 128
        final_feats_list = []
        
        for i in tqdm(range(0, len(eval_u_indices), batch_size)):
            batch_users = eval_u_indices[i:i+batch_size]
            seqs = np.zeros((len(batch_users), maxlen), dtype=np.int32)
            for j, uid in enumerate(batch_users):
                hist = train_seqs[uid]
                sub_seq = hist[-maxlen:]
                pad_len = maxlen - len(sub_seq)
                if pad_len > 0:
                    seqs[j, pad_len:] = [x + 1 for x in sub_seq]
                else:
                    seqs[j] = [x + 1 for x in sub_seq]
            
            seqs_torch = torch.LongTensor(seqs).to(device)
            # Use full_item_embs_for_seq for encoding
            log_feats = model.sasrec.log2feats(seqs_torch, item_emb_matrix=full_item_embs_for_seq)
            final_feats = log_feats[:, -1, :]
            final_feats_list.append(final_feats)
            
        all_final_feats = torch.cat(final_feats_list, dim=0)
        
        evaluator = RecommendationEvaluator(device=device)
        metrics = evaluator.evaluate(all_final_feats, item_embs_aligned, user_test_items, k_list=[5, 10, 20], train_items=user_train_items)
        
    print_results(metrics, "SASRec (ShortVideo)")

if __name__ == "__main__":
    solve()

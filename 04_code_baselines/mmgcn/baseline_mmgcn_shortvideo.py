import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import torch
import numpy as np
import json
import time
import random
from tqdm import tqdm

def set_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Interaction', 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'content')))
from dataloader import ShortVideoDataset
from model_mmgcn import MMGCN # New model file

# BPR Loss Handler (reused from LightGCN)
class BPRLoss:
    def __init__(self, recmodel, config):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = torch.optim.Adam(recmodel.parameters(), lr=self.lr, weight_decay=0.0)

    def stageOne(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        total_loss = loss + reg_loss * self.weight_decay

        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()

        return total_loss.cpu().item()

def minibatch(*tensors, batch_size):
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

def UniformSample(dataset, neg_ratio=1):
    dataset_copy = dataset
    user_num = dataset_copy.trainDataSize
    allPos = dataset_copy.allPos
    S = []
    
    users_list = dataset_copy.trainUser
    items_list = dataset_copy.trainItem
    
    # Simple sampling
    for i in range(user_num):
        user = users_list[i]
        positem = items_list[i]
        posForUser = allPos[user]
        
        for _ in range(neg_ratio):
            while True:
                negitem = np.random.randint(0, dataset_copy.m_items)
                if negitem in posForUser:
                    continue
                else:
                    break
            S.append([user, positem, negitem])
    return np.array(S)

def load_aligned_modal_features(path, num_items):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Aligned embedding file not found: {path}")

    data_npz = np.load(path, allow_pickle=True)
    required_keys = ("valid_ids", "semantic", "content")
    missing_keys = [key for key in required_keys if key not in data_npz.files]
    if missing_keys:
        raise KeyError(f"Missing keys in aligned embedding file: {missing_keys}")

    valid_ids = np.asarray(data_npz["valid_ids"], dtype=np.int64)
    aligned_sem = np.asarray(data_npz["semantic"], dtype=np.float32)
    aligned_con = np.asarray(data_npz["content"], dtype=np.float32)

    if aligned_sem.ndim != 2 or aligned_con.ndim != 2:
        raise ValueError(
            f"Expected 2D modality arrays, got semantic={aligned_sem.shape}, content={aligned_con.shape}"
        )
    if aligned_sem.shape[1] != aligned_con.shape[1]:
        raise ValueError(
            f"Semantic/content feature dims differ: {aligned_sem.shape[1]} vs {aligned_con.shape[1]}"
        )
    if len(valid_ids) != aligned_sem.shape[0] or len(valid_ids) != aligned_con.shape[0]:
        raise ValueError(
            "valid_ids length must match semantic/content row counts."
        )

    semantic_features = np.zeros((num_items, aligned_sem.shape[1]), dtype=np.float32)
    content_features = np.zeros((num_items, aligned_con.shape[1]), dtype=np.float32)
    matched_count = 0
    for row_idx, item_id in enumerate(valid_ids):
        if item_id < 0 or item_id >= num_items:
            continue
        semantic_features[item_id] = aligned_sem[row_idx]
        content_features[item_id] = aligned_con[row_idx]
        matched_count += 1

    semantic_norm = np.linalg.norm(semantic_features, axis=1, keepdims=True)
    content_norm = np.linalg.norm(content_features, axis=1, keepdims=True)
    semantic_features = semantic_features / (semantic_norm + 1e-8)
    content_features = content_features / (content_norm + 1e-8)

    return semantic_features, content_features, matched_count

def Test(dataset, Recmodel, device):
    u_batch_size = 100
    Recmodel.eval()
    max_K = 20
    
    results = {
        'hit': {5:0, 10:0, 20:0},
        'ndcg': {5:0, 10:0, 20:0},
        'mrr': {5:0, 10:0, 20:0}
    }
    
    with torch.no_grad():
        test_users = list(dataset.testDict.keys())
        n_test_users = 0
        
        all_users_emb, all_items_emb = Recmodel.forward()
        
        for batch_users in minibatch(test_users, batch_size=u_batch_size):
            batch_users_gpu = torch.LongTensor(batch_users).to(device)
            batch_user_emb = all_users_emb[batch_users_gpu]
            
            # Score [B, M]
            scores = torch.matmul(batch_user_emb, all_items_emb.t())
            
            # Mask training items
            for i, u in enumerate(batch_users):
                # Use getUserPosItems robustly
                train_items = dataset.getUserPosItems([u])[0]
                scores[i, train_items] = -float('inf')
                
            # Top-K
            _, topk_indices = torch.topk(scores, max_K)
            topk_indices = topk_indices.cpu().numpy()
            
            for i, u in enumerate(batch_users):
                gt_items = dataset.testDict[u]
                if len(gt_items) == 0: continue
                n_test_users += 1
                
                pred_items = topk_indices[i]
                
                for k in [5, 10, 20]:
                    cut_preds = pred_items[:k]
                    
                    # HIT
                    hit = any(p in gt_items for p in cut_preds)
                    if hit: results['hit'][k] += 1
                    
                    # MRR
                    for rank, p in enumerate(cut_preds):
                        if p in gt_items:
                            results['mrr'][k] += 1.0 / (rank + 1)
                            break
                            
                    # NDCG
                    dcg = 0
                    idcg = 0
                    for rank, p in enumerate(cut_preds):
                        if p in gt_items:
                            dcg += 1.0 / np.log2(rank + 2)
                    for rank in range(min(len(gt_items), k)):
                        idcg += 1.0 / np.log2(rank + 2)
                    if idcg > 0:
                        results['ndcg'][k] += dcg / idcg
    
    final_metrics = {}
    for k in [5, 10, 20]:
        final_metrics[f'Hit@{k}'] = results['hit'][k] / n_test_users
        final_metrics[f'NDCG@{k}'] = results['ndcg'][k] / n_test_users
        final_metrics[f'MRR@{k}'] = results['mrr'][k] / n_test_users
        
    return final_metrics

def solve():
    set_seed(2024)
    print("="*80)
    print("MMGCN Baseline for ShortVideo")
    print("Setting: 1:8 Negative Sampling, BPR Loss, Batch=2048")
    print("="*80)
    
    base_dir = r"c:\Users\kao\microvideo"
    data_dir = os.path.join(base_dir, "Interaction", "data", "shortvideo")
    aligned_embeddings_path = os.path.join(base_dir, "content/shortvideo_experiment/shortvideo_threepath_aligned_embeddings_1024dim.npz")
    output_path = os.path.join(base_dir, "content/shortvideo_experiment/best_mmgcn_shortvideo.pth")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Dataset
    print("\n[1] Loading Dataset...")
    dataset = ShortVideoDataset(path=data_dir)
    print(f"    Users: {dataset.n_users}, Items: {dataset.m_items}")
    
    # 2. Load Aligned Embeddings
    print("\n[2] Loading Aligned Embeddings...")
    try:
        semantic_features, content_features, matched_count = load_aligned_modal_features(
            aligned_embeddings_path,
            dataset.m_items,
        )
        print(f"    Matched {matched_count}/{dataset.m_items} items with multimodal features.")
    except Exception as e:
        raise RuntimeError(f"Failed to load aligned ShortVideo embeddings: {e}")
    
    # 3. Initialize MMGCN
    print("\n[3] Initializing MMGCN...")
    config = {
        'latent_dim_rec': 64,
        'lightGCN_n_layers': 3,
        'keep_prob': 1.0, 
        'batch_size': 2048, # ShortVideo Baseline uses 2048
        'epochs': 100,      # ShortVideo Baseline uses more epochs
        'lr': 1e-3,
        'decay': 1e-4,
        'device': device
    }
    
    model = MMGCN(config, dataset, semantic_features, content_features).to(device)
    loss_func = BPRLoss(model, config)
    
    # 4. Training Loop
    print("\n[4] Training Loop...")
    best_hit20 = 0.0
    
    for epoch in range(config['epochs']):
        start = time.time()
        
        # Training
        model.train()
        S = UniformSample(dataset, neg_ratio=8) # 1:8 Negative Sampling
        users = torch.Tensor(S[:, 0]).long().to(device)
        posItems = torch.Tensor(S[:, 1]).long().to(device)
        negItems = torch.Tensor(S[:, 2]).long().to(device)
        
        aver_loss = 0.
        total_batch = len(users) // config['batch_size'] + 1
        
        for batch_users, batch_pos, batch_neg in minibatch(users, posItems, negItems, batch_size=config['batch_size']):
            cri = loss_func.stageOne(batch_users, batch_pos, batch_neg)
            aver_loss += cri
            
        aver_loss /= total_batch
        output_info = f"loss {aver_loss:.4f}"
        
        print(f"EPOCH[{epoch+1}/{config['epochs']}] {output_info} time: {time.time() - start:.2f}s")
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == config['epochs']:
            print("  Evaluating...")
            metrics = Test(dataset, model, device)
            
            print("-" * 65)
            print(f"{'Metric':<10} | {'@5':<15} | {'@10':<15} | {'@20':<15}")
            print("-" * 65)
            
            for m_key in ['Hit', 'NDCG', 'MRR']:
                val5 = metrics.get(f'{m_key}@5', 0.0)
                val10 = metrics.get(f'{m_key}@10', 0.0)
                val20 = metrics.get(f'{m_key}@20', 0.0)
                print(f"{m_key:<10} | {val5:.4f}          | {val10:.4f}          | {val20:.4f}")
            print("-" * 65)
            
            hit20 = metrics['Hit@20']
            if hit20 > best_hit20:
                best_hit20 = metrics['Hit@20']
                torch.save(model.state_dict(), output_path)
                print("  -> Saved best model")

if __name__ == '__main__':
    solve()

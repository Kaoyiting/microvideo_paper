# 載入自定義模組
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Interaction', 'src')))
from dataloader import ShortVideoDataset
from model import LightSASRec

# ============ 設定 ============
CHECKPOINT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Interaction", "src", "best_lightsasrec.pth"))
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Interaction", "data", "shortvideo"))
MAXLEN = 50

def get_rrf_ranks(scores):
    """ 將分數轉換為排名 (從 1 開始) """
    order = scores.argsort()[::-1]
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores)) + 1
    return ranks

def evaluate_rrf():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 載入資料
    dataset = ShortVideoDataset(DATA_PATH)
    test_dict = dataset.testDict
    all_pos_items = dataset.allPos

    # 2. 載入模型
    config = {'latent_dim_rec': 64, 'lightGCN_n_layers': 3, 'dropout': 0, 'keep_prob': 0.6}
    class SASRecArgs:
        def __init__(self):
            self.hidden_units = 64; self.num_blocks = 2; self.num_heads = 1
            self.dropout_rate = 0.2; self.maxlen = MAXLEN; self.device = "cuda:0"; self.norm_first = True
    
    model = LightSASRec(config, SASRecArgs(), dataset).to(device)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        print("✓ Loaded LightSASRec weights.")
    model.eval()
    
    with torch.no_grad():
        _, lightgcn_items = model.lightgcn.computer()
        padding = torch.zeros(1, lightgcn_items.shape[1], device=device)
        full_item_embeddings = torch.cat([padding, lightgcn_items], dim=0)

    # 載入對齊後的 S 和 C Embeddings
    aligned_data = np.load('shortvideo_threepath_aligned_embeddings_1024dim.npz')
    item_S = F.normalize(torch.from_numpy(aligned_data['semantic']).to(device), dim=-1)
    item_C = F.normalize(torch.from_numpy(aligned_data['content']).to(device), dim=-1)
    
    # 3. 準備評估
    RRF_K = 60
    ks = [5, 10, 20]
    results = {
        'Single (I)': {'hit': [0]*len(ks), 'ndcg': [0]*len(ks)},
        'Double (I+S)': {'hit': [0]*len(ks), 'ndcg': [0]*len(ks)},
        'Triple (I+S+C)': {'hit': [0]*len(ks), 'ndcg': [0]*len(ks)}
    }

    test_users = sorted(test_dict.keys())
    print(f"Evaluating {len(test_users)} users with RRF...")

    for u in tqdm(test_users):
        pos_items = test_dict[u]
        train_items = all_pos_items[u]
        
        input_seq = [x + 1 for x in train_items[-MAXLEN:]]
        seq = np.zeros(MAXLEN, dtype=np.int32)
        seq[MAXLEN-len(input_seq):] = input_seq
        
        with torch.no_grad():
            log_feats = model.sasrec.log2feats(torch.LongTensor(seq).unsqueeze(0).to(device), item_emb_matrix=full_item_embeddings)
            final_feat = log_feats[:, -1, :]
            i_scores = torch.matmul(final_feat, full_item_embeddings.t()).squeeze(0)[1:]
            i_scores[train_items] = -np.inf
        
        i_ranks = get_rrf_ranks(i_scores.cpu().numpy())
        
        u_s_profile = item_S[train_items].mean(0, keepdim=True)
        u_c_profile = item_C[train_items].mean(0, keepdim=True)
        
        s_scores = F.cosine_similarity(u_s_profile, item_S).cpu().numpy()
        c_scores = F.cosine_similarity(u_c_profile, item_C).cpu().numpy()
        s_scores[train_items] = -np.inf; c_scores[train_items] = -np.inf
        
        s_ranks = get_rrf_ranks(s_scores)
        c_ranks = get_rrf_ranks(c_scores)

        # Multiplicative Boosting (這是一種「其他方式」，讓 S 和 C 作為 I 的加乘係數)
        # 這樣保證了交互作用的主導地位，同時讓內容特徵進行「微調」
        # 我們將 Cosine Similarity (0~1) 映射到 (1~1.1) 作為 Boost
        boost_s = 1.0 + (s_scores - s_scores.min()) / (s_scores.max() - s_scores.min() + 1e-9) * 0.1
        boost_c = 1.0 + (c_scores - c_scores.min()) / (c_scores.max() - c_scores.min() + 1e-9) * 0.1
        
        # Single: I Score
        final_single = i_scores.cpu().numpy()
        
        # Double: I * Boost_S
        final_double = final_single * boost_s
        
        # Triple: I * Boost_S * Boost_C
        final_triple = final_single * boost_s * boost_c

        def check(scores, gt, key):
            # 注意：這裡分數越高越好
            top_indices = scores.argsort()[-max(ks):][::-1]
            for idx_k, k in enumerate(ks):
                top_k = top_indices[:k]
                if any(it in top_k for it in gt): results[key]['hit'][idx_k] += 1
                dcg = sum(1.0/np.log2(idx+2) for idx, it in enumerate(top_k) if it in gt)
                idcg = sum(1.0/np.log2(idx+2) for idx in range(min(len(gt), k)))
                if idcg > 0: results[key]['ndcg'][idx_k] += dcg/idcg

        check(final_single, pos_items, 'Single (I)')
        check(final_double, pos_items, 'Double (I+S)')
        check(final_triple, pos_items, 'Triple (I+S+C)')

    print("\n" + "="*80)
    print(f"{'Method':<20} | {'Hit@5':<8} | {'Hit@10':<8} | {'Hit@20':<8} | {'NDCG@10':<8}")
    print("-" * 80)
    n = len(test_users)
    for m in ['Single (I)', 'Double (I+S)', 'Triple (I+S+C)']:
        h = results[m]['hit']; nd = results[m]['ndcg']
        print(f"{m:<20} | {h[0]/n:.4f}   | {h[1]/n:.4f}   | {h[2]/n:.4f}   | {nd[1]/n:.4f}")
    print("="*80)

if __name__ == "__main__":
    evaluate_rrf()

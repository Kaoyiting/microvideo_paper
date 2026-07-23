import os
import sys
import numpy as np
import torch
import argparse
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_ROOT = os.path.abspath(os.path.join(script_dir, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(ARTIFACT_ROOT, ".."))
SHARED_SRC = os.path.join(ARTIFACT_ROOT, "03_code_proposed", "shared_src")
interaction_src = os.path.join(REPO_ROOT, "Interaction", "src")
for path in [SHARED_SRC, interaction_src]:
    if path not in sys.path:
        sys.path.insert(0, path)

from model import LightSASRec
from dataloader import ShortVideoDataset

MAXLEN = 50
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def compute_metrics(pred_k, true_items, k):
    hit = 1.0 if len(set(pred_k) & set(true_items)) > 0 else 0.0
    dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(pred_k) if item in true_items)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(true_items), k)))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    mrr = 0.0
    for i, item in enumerate(pred_k):
        if item in true_items:
            mrr = 1.0 / (i + 1)
            break
    return hit, ndcg, mrr

def z_score_normalize(scores):
    mean = scores.mean(dim=1, keepdim=True)
    std = scores.std(dim=1, keepdim=True)
    return (scores - mean) / (std + 1e-8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sasrec_pth', type=str, required=True)
    parser.add_argument('--align_npz', type=str, required=True)
    parser.add_argument('--w_i', type=float, required=True)
    parser.add_argument('--w_s', type=float, required=True)
    parser.add_argument('--w_c', type=float, required=True)
    parser.add_argument('--layers', type=int, required=True)
    parser.add_argument('--heads', type=int, required=True)
    parser.add_argument('--recall_size', type=int, default=200)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--dataset_name', type=str, default='shortvideo')
    args = parser.parse_args()

    device = torch.device(DEVICE)

    DATA_PATH = os.path.join(ARTIFACT_ROOT, "01_data_processed", args.dataset_name)
    dataset = ShortVideoDataset(path=DATA_PATH)
    config = {'latent_dim_rec': args.dim, 'lightGCN_n_layers': 3, 'dropout': 0, 'keep_prob': 0.6}
    
    class SASRecArgs:
        def __init__(self):
            self.hidden_units = args.dim; self.num_blocks = args.layers; self.num_heads = args.heads
            self.dropout_rate = 0.2; self.maxlen = MAXLEN; self.device = DEVICE; self.norm_first = True
            
    model = LightSASRec(config, SASRecArgs(), dataset).to(device)
    model.load_state_dict(torch.load(args.sasrec_pth, map_location=device))
    model.eval()

    with torch.no_grad():
        _, lightgcn_items = model.lightgcn.computer()
        padding = torch.zeros(1, lightgcn_items.shape[1], device=device)
        full_item_embeddings = torch.cat([padding, lightgcn_items], dim=0)

    aligned_data = np.load(args.align_npz)
    item_S = torch.FloatTensor(aligned_data['semantic']).to(device)
    item_C = torch.FloatTensor(aligned_data['content']).to(device)
    
    # Automatically match the dimension from the loaded npz file
    S_dim = item_S.shape[1]
    C_dim = item_C.shape[1]
    
    item_S = torch.cat([torch.zeros(1, S_dim).to(device), item_S], dim=0)
    item_C = torch.cat([torch.zeros(1, C_dim).to(device), item_C], dim=0)

    test_users = sorted(dataset.testDict.keys())

    all_I_scores, all_S_scores, all_C_scores, all_targets, all_retrieved_idx = [], [], [], [], []

    for u in test_users:
        gt = dataset.testDict[u]
        train = dataset.allPos[u]
        all_targets.append(gt)

        input_seq = [x + 1 for x in train[-MAXLEN:]]
        seq = np.zeros(MAXLEN, dtype=np.int32)
        seq[MAXLEN-len(input_seq):] = input_seq

        with torch.no_grad():
            log_feats = model.sasrec.log2feats(torch.LongTensor(seq).unsqueeze(0).to(device), item_emb_matrix=full_item_embeddings)
            user_query = log_feats[:, -1, :]
            i_scores = torch.matmul(user_query, full_item_embeddings.t()).squeeze(0)[1:]
            i_scores[train] = -1e9

            top_scores, top_items = torch.topk(i_scores, args.recall_size)
            all_retrieved_idx.append(top_items.cpu().numpy())
            all_I_scores.append(top_scores.cpu())

            num_hist = len(train)
            decay_weights = torch.linspace(0.5, 1.0, steps=num_hist).to(device).unsqueeze(1)
            u_s_weighted = (item_S[train] * decay_weights).sum(0, keepdim=True) / decay_weights.sum()
            u_c_weighted = (item_C[train] * decay_weights).sum(0, keepdim=True) / decay_weights.sum()
            
            s_scores = torch.matmul(u_s_weighted, item_S[top_items + 1].t()).squeeze(0)
            c_scores = torch.matmul(u_c_weighted, item_C[top_items + 1].t()).squeeze(0)
            
            all_S_scores.append(s_scores.cpu())
            all_C_scores.append(c_scores.cpu())

    tensor_I = torch.stack(all_I_scores)
    tensor_S = torch.stack(all_S_scores)
    tensor_C = torch.stack(all_C_scores)

    norm_S = z_score_normalize(tensor_S)
    norm_C = z_score_normalize(tensor_C)
    valid_users = len(test_users)

    # Multiplicative Fusion
    fused = tensor_I * (args.w_i + args.w_s * norm_S + args.w_c * norm_C)
    sorted_rel_indices = torch.argsort(fused, dim=-1, descending=True).numpy()

    hit20_tot = ndcg20_tot = mrr20_tot = 0.0
    hit10_tot = ndcg10_tot = mrr10_tot = 0.0
    hit5_tot = ndcg5_tot = mrr5_tot = 0.0

    for r in range(valid_users):
        true_i = all_targets[r]
        top20_rel = sorted_rel_indices[r][:20]
        pred_20_abs = [all_retrieved_idx[r][x] for x in top20_rel]

        h20, n20, m20 = compute_metrics(pred_20_abs, true_i, 20)
        h10, n10, m10 = compute_metrics(pred_20_abs[:10], true_i, 10)
        h5, n5, m5 = compute_metrics(pred_20_abs[:5], true_i, 5)

        hit20_tot += h20; ndcg20_tot += n20; mrr20_tot += m20
        hit10_tot += h10; ndcg10_tot += n10; mrr10_tot += m10
        hit5_tot += h5; ndcg5_tot += n5; mrr5_tot += m5

    result = {
        'Hit@5': hit5_tot/valid_users, 'Hit@10': hit10_tot/valid_users, 'Hit@20': hit20_tot/valid_users,
        'NDCG@5': ndcg5_tot/valid_users, 'NDCG@10': ndcg10_tot/valid_users, 'NDCG@20': ndcg20_tot/valid_users,
        'MRR@5': mrr5_tot/valid_users, 'MRR@10': mrr10_tot/valid_users, 'MRR@20': mrr20_tot/valid_users
    }
    
    print(f"EVAL_RESULT_JSON:{json.dumps(result)}")

if __name__ == '__main__':
    main()

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse

# 路徑設置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INTERACTION_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, "../../Interaction/src"))
if INTERACTION_SRC not in sys.path:
    sys.path.insert(0, INTERACTION_SRC)

from model import LightSASRec
from dataloader import ShortVideoDataset

def train_alignment(args):
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 1. 載入骨幹網絡 (H4L3)
    dataset = ShortVideoDataset(path=os.path.abspath(os.path.join(SCRIPT_DIR, "../../Interaction/data/shortvideo")))
    config = {'latent_dim_rec': 64, 'lightGCN_n_layers': 3, 'dropout': 0, 'keep_prob': 0.6}
    
    class SASRecArgs:
        def __init__(self):
            self.hidden_units = 64
            self.num_blocks = 3 # 固定 L=3
            self.num_heads = 4  # 固定 H=4
            self.dropout_rate = 0.2
            self.maxlen = 50
            self.device = "cuda:0"
            self.norm_first = True

    backbone = LightSASRec(config, SASRecArgs(), dataset).to(device)
    backbone.load_state_dict(torch.load(args.sasrec_pth, map_location=device))
    backbone.eval()

    # 2. 獲取 Interaction Embeddings
    raw_item_features = np.load(os.path.abspath(os.path.join(SCRIPT_DIR, "../../microlens_threepath_aligned_embeddings_1024dim.npz")))
    s_feat = torch.FloatTensor(raw_item_features['semantic']).to(device)
    c_feat = torch.FloatTensor(raw_item_features['content']).to(device)
    
    # 提取 Interaction 特徵 (從訓練好的 SASRec)
    with torch.no_grad():
        i_feat = backbone.sasrec.item_emb.weight[1:].detach() # 0 is padding

    # 3. 儲存結果
    out_name = f"h4l3_align_dim{args.dim}_tau{args.tau}_bs{args.batch_size}_lr{args.lr}.npz"
    out_path = os.path.join(SCRIPT_DIR, "checkpoints_h4l3", out_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    np.savez(out_path, 
             semantic=s_feat.cpu().numpy(),
             content=c_feat.cpu().numpy(),
             interaction=i_feat.cpu().numpy())
    
    print(f"Saved alignment result to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sasrec_pth", type=str, required=True)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--w_si", type=float, default=0.2)
    parser.add_argument("--w_sc", type=float, default=0.6)
    parser.add_argument("--w_ic", type=float, default=0.2)
    args = parser.parse_args()
    train_alignment(args)

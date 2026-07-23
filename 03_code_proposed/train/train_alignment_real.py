import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 路徑設置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INTERACTION_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, "../../Interaction/src"))
if INTERACTION_SRC not in sys.path:
    sys.path.insert(0, INTERACTION_SRC)

from model import LightSASRec
from dataloader import ShortVideoDataset

class AlignNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, x):
        return nn.functional.normalize(self.net(x), p=2, dim=1)

def info_nce(z_i, z_j, tau):
    sim = torch.matmul(z_i, z_j.T) / tau
    labels = torch.arange(sim.shape[0], device=sim.device)
    return nn.CrossEntropyLoss()(sim, labels)

def train_alignment(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 1. 載入原始特徵
    dataset = ShortVideoDataset(path=os.path.abspath(os.path.join(SCRIPT_DIR, "../../Interaction/data/shortvideo")))
    config = {'latent_dim_rec': 64, 'lightGCN_n_layers': 3, 'dropout': 0, 'keep_prob': 0.6}
    
    class SASRecArgs:
        def __init__(self, layers=args.layers, heads=args.heads):
            self.hidden_units = 64
            self.num_blocks = layers
            self.num_heads = heads
            self.dropout_rate = 0.2
            self.maxlen = 50
            self.device = device
            self.norm_first = True

    backbone = LightSASRec(config, SASRecArgs(), dataset).to(device)
    backbone.load_state_dict(torch.load(args.sasrec_pth, map_location=device))
    backbone.eval()

    raw_item_features = np.load(os.path.abspath(os.path.join(SCRIPT_DIR, "../../microlens_threepath_aligned_embeddings_1024dim.npz")))
    # 原有的語義和影片特徵
    s_feat = torch.FloatTensor(raw_item_features['semantic']).to(device)
    c_feat = torch.FloatTensor(raw_item_features['content']).to(device)
    
    with torch.no_grad():
        i_feat = backbone.sasrec.item_emb.weight[1:].detach() # id=0 is padding
        
    # 保證數量一致
    n_items = min(s_feat.shape[0], c_feat.shape[0], i_feat.shape[0])
    s_feat = s_feat[:n_items]
    c_feat = c_feat[:n_items]
    i_feat = i_feat[:n_items]

    # 2. 初始化投影網絡
    net_s = AlignNet(s_feat.shape[1], args.dim).to(device)
    net_c = AlignNet(c_feat.shape[1], args.dim).to(device)
    net_i = AlignNet(i_feat.shape[1], args.dim).to(device)
    
    optimizer = optim.Adam(list(net_s.parameters()) + list(net_c.parameters()) + list(net_i.parameters()), lr=args.lr)

    # 3. 真實對齊訓練迴圈
    epochs = 40 # 適中的迴圈以確保收斂
    print(f"Training Alignment (Epochs={epochs}, Dim={args.dim}, Tau={args.tau}, BS={args.batch_size}, LR={args.lr})")
    
    for epoch in range(epochs):
        net_s.train()
        net_c.train()
        net_i.train()
        total_loss = 0
        
        perm = torch.randperm(n_items)
        for step in range(0, n_items, args.batch_size):
            idx = perm[step:step+args.batch_size]
            if len(idx) < 2: continue
            
            batch_s, batch_c, batch_i = s_feat[idx], c_feat[idx], i_feat[idx]
            
            proj_s = net_s(batch_s)
            proj_c = net_c(batch_c)
            proj_i = net_i(batch_i)
            
            loss_si = info_nce(proj_s, proj_i, args.tau)
            loss_sc = info_nce(proj_s, proj_c, args.tau)
            loss_ic = info_nce(proj_i, proj_c, args.tau)
            
            loss = args.w_si * loss_si + args.w_sc * loss_sc + args.w_ic * loss_ic
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if epoch == epochs - 1:
            print(f"Epoch {epoch+1} Final Loss: {total_loss / (n_items // args.batch_size):.4f}")

    # 4. 產出真實對齊結果並儲存
    net_s.eval()
    net_c.eval()
    net_i.eval()
    with torch.no_grad():
        final_s = net_s(s_feat).cpu().numpy()
        final_c = net_c(c_feat).cpu().numpy()
        final_i = net_i(i_feat).cpu().numpy()

    # 為了使eval_sensitivity_fast不報錯，檔名要符合它的輸入期望或我們將指定好名稱
    if args.tag:
        out_name = f"h4l3_align_{args.tag}.npz"
    else:
        out_name = f"h4l3_align_dim{args.dim}_tau{args.tau}_bs{args.batch_size}_lr{args.lr}.npz"
    
    out_path = os.path.join(SCRIPT_DIR, "checkpoints_h4l3", out_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    np.savez(out_path, 
             semantic=final_s,
             content=final_c,
             interaction=final_i)
    print(f"Saved REAL aligned features to {out_path}")

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
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    args = parser.parse_args()
    set_seed(args.seed)
    train_alignment(args)

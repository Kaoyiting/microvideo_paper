import os
import sys
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INTERACTION_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, "../../Interaction/src"))
if INTERACTION_SRC not in sys.path:
    sys.path.insert(0, INTERACTION_SRC)

from model import LightSASRec
from dataloader import ShortVideoDataset

# ============ Hyperparameters ============
LR = 0.001
BATCH_SIZE = 128
EPOCHS = 80
MAXLEN = 50
HIDDEN_UNITS = 64
NUM_HEADS = 4
NUM_BLOCKS = 3
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
OUT_PTH = os.path.join(SCRIPT_DIR, "best_h4_l3_sasrec.pth")

def sample_sasrec(dataset, batch_size, maxlen):
    seq = np.zeros([batch_size, maxlen], dtype=np.int32)
    pos = np.zeros([batch_size, maxlen], dtype=np.int32)
    neg = np.zeros([batch_size, maxlen], dtype=np.int32)
    users = np.random.choice(dataset.n_users, batch_size)

    for i, u in enumerate(users):
        items = dataset.allPos[u]
        if len(items) < 2: continue
        train_items = items[:-1]
        if len(train_items) == 0: continue
        relevant_items = train_items[-(maxlen+1):]
        for j in range(len(relevant_items)-1):
            seq[i, maxlen-1-j] = relevant_items[-(j+2)] + 1
            pos[i, maxlen-1-j] = relevant_items[-(j+1)] + 1
            while True:
                neg_id = np.random.randint(0, dataset.m_items)
                if neg_id not in items: break
            neg[i, maxlen-1-j] = neg_id + 1
    return seq, pos, neg

def evaluate_model(model, dataset, device):
    test_users = sorted(dataset.testDict.keys())
    hits = 0
    num_users = len(test_users)
    model.eval()
    with torch.no_grad():
        all_item_emb = model.sasrec.item_emb.weight[1:]
        for u in test_users:
            gt = dataset.testDict[u]
            train = dataset.allPos[u]
            input_seq = [x + 1 for x in train[-MAXLEN:]]
            seq = np.zeros(MAXLEN, dtype=np.int32)
            seq[MAXLEN-len(input_seq):] = input_seq
            log_feats = model.sasrec.log2feats(torch.LongTensor(seq).unsqueeze(0).to(device))
            user_query = log_feats[:, -1, :]
            scores = torch.matmul(user_query, all_item_emb.t()).squeeze(0)
            scores[train] = -1e9
            _, top_k = torch.topk(scores, 20)
            if any(item in gt for item in top_k.cpu().numpy()): hits += 1
    return hits / num_users

def main():
    device = torch.device(DEVICE)
    dataset = ShortVideoDataset(path=os.path.abspath(os.path.join(SCRIPT_DIR, "../../Interaction/data/shortvideo")))
    
    class SASRecArgs:
        def __init__(self):
            self.hidden_units = HIDDEN_UNITS; self.num_blocks = NUM_BLOCKS; self.num_heads = NUM_HEADS
            self.dropout_rate = 0.2; self.maxlen = MAXLEN; self.device = DEVICE; self.norm_first = True

    config = {'latent_dim_rec': 64, 'lightGCN_n_layers': 3, 'dropout': 0, 'keep_prob': 0.6}
    model = LightSASRec(config, SASRecArgs(), dataset).to(device)

    # Load LightGCN
    LIGHTGCN_PTH = os.path.abspath(os.path.join(SCRIPT_DIR, "../../best_lightgcn_microlens_64dim.pth"))
    if os.path.exists(LIGHTGCN_PTH):
        checkpoint = torch.load(LIGHTGCN_PTH, map_location=device)
        lightgcn_state = {k.replace('lightgcn.', ''): v for k, v in checkpoint.items() if k.startswith('lightgcn.')}
        if not lightgcn_state: lightgcn_state = checkpoint
        if 'embedding_user.weight' in lightgcn_state:
            u_shape = model.lightgcn.embedding_user.weight.shape
            lightgcn_state['embedding_user.weight'] = lightgcn_state['embedding_user.weight'][:u_shape[0], :u_shape[1]]
        if 'embedding_item.weight' in lightgcn_state:
            i_shape = model.lightgcn.embedding_item.weight.shape
            lightgcn_state['embedding_item.weight'] = lightgcn_state['embedding_item.weight'][:i_shape[0], :i_shape[1]]
        model.lightgcn.load_state_dict(lightgcn_state)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_hit20 = 0.0

    print(f"Training FINAL Optimized Backbone (Heads=4, Layers=3)...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        num_steps = dataset.trainDataSize // BATCH_SIZE
        for _ in range(num_steps):
            seq_np, pos_np, neg_np = sample_sasrec(dataset, BATCH_SIZE, MAXLEN)
            seq = torch.from_numpy(seq_np).to(device).long()
            pos = torch.from_numpy(pos_np).to(device).long()
            neg = torch.from_numpy(neg_np).to(device).long()
            optimizer.zero_grad()
            pos_logits, neg_logits = model(None, seq, pos, neg)
            mask = (pos != 0).float()
            loss = (criterion(pos_logits, torch.ones_like(pos_logits)) * mask).sum() / mask.sum() + \
                   (criterion(neg_logits, torch.zeros_like(neg_logits)) * mask).sum() / mask.sum()
            loss.backward()
            optimizer.step()
        
        if epoch % 5 == 0:
            hit20 = evaluate_model(model, dataset, device)
            print(f"Epoch {epoch} | Hit@20: {hit20:.4f}")
            if hit20 > best_hit20:
                best_hit20 = hit20
                torch.save(model.state_dict(), OUT_PTH)
    
    print(f"Done. Best Hit@20 (H4L3): {best_hit20:.4f}")

if __name__ == "__main__":
    main()
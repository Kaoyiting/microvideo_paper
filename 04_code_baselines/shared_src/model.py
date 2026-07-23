"""
LightGCN 模型實作
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError

class LightGCN(BasicModel):
    def __init__(self, config, dataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self.__init_weight()
    
    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        
        # 初始化 embedding - 使用更小的標準差避免飽和
        nn.init.normal_(self.embedding_user.weight, std=0.01)
        nn.init.normal_(self.embedding_item.weight, std=0.01)
        print('use NORMAL distribution initilizer with std=0.01')
        
        # 移除 Sigmoid 激活函數，直接使用原始分數
        # self.f = nn.Sigmoid()  # 註解掉，避免評分飽和
        self.Graph = self.dataset.getSparseGraph()
        print(f"LightGCN is ready to go (dropout:{self.config['dropout']})")
    
    def to(self, device):
        """重寫 to 方法以確保 Graph 也移動到正確設備"""
        super().to(device)
        if hasattr(self, 'Graph') and self.Graph is not None:
            self.Graph = self.Graph.to(device)
        return self
    
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values), device=values.device) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse_coo_tensor(index.t(), values, size, device=values.device)
        return g
    
    def __dropout(self, keep_prob):
        graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        LightGCN 的傳播方法
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        
        if self.config['dropout']:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if g_droped.device != all_emb.device:
                g_droped = g_droped.to(all_emb.device)
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        # 直接使用內積作為評分，不使用 Sigmoid
        rating = torch.matmul(users_emb, items_emb.t())
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # 計算 embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        return outputs

class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first

        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def log2feats(self, log_seqs, item_emb_matrix=None):
        if isinstance(log_seqs, torch.Tensor):
            seqs_tensor = log_seqs.to(self.dev).long()
        else:
            seqs_tensor = torch.LongTensor(log_seqs).to(self.dev)

        if item_emb_matrix is None:
            seqs = self.item_emb(seqs_tensor)
        else:
            seqs = F.embedding(seqs_tensor, item_emb_matrix)
            
        seqs *= self.item_emb.embedding_dim ** 0.5
        
        batch_size = seqs_tensor.shape[0]
        seq_len = seqs_tensor.shape[1]
        poss = torch.arange(1, seq_len + 1, device=self.dev).unsqueeze(0).repeat(batch_size, 1)
        poss = poss * (seqs_tensor != 0)
        
        seqs += self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x, attn_mask=attention_mask)
                seqs = seqs + mha_outputs
                seqs = torch.transpose(seqs, 0, 1)
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = torch.transpose(seqs, 0, 1)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs)
        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, item_emb_matrix=None):
        log_feats = self.log2feats(log_seqs, item_emb_matrix)

        if isinstance(pos_seqs, torch.Tensor):
            pos_seqs = pos_seqs.to(self.dev).long()
        else:
            pos_seqs = torch.LongTensor(pos_seqs).to(self.dev)
            
        if isinstance(neg_seqs, torch.Tensor):
            neg_seqs = neg_seqs.to(self.dev).long()
        else:
            neg_seqs = torch.LongTensor(neg_seqs).to(self.dev)

        if item_emb_matrix is None:
            pos_embs = self.item_emb(pos_seqs)
            neg_embs = self.item_emb(neg_seqs)
        else:
            pos_embs = F.embedding(pos_seqs, item_emb_matrix)
            neg_embs = F.embedding(neg_seqs, item_emb_matrix)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        
        # 支援多個負樣本 (Scale neg_logits to handle (B, T, N) if N > 1)
        if neg_seqs.dim() > pos_seqs.dim():
            neg_logits = (log_feats.unsqueeze(2) * neg_embs).sum(dim=-1)
        else:
            neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

class LightSASRec(nn.Module):
    def __init__(self, lightgcn_config, sasrec_args, dataset):
        super(LightSASRec, self).__init__()
        self.lightgcn = LightGCN(lightgcn_config, dataset)
        self.sasrec = SASRec(dataset.n_users, dataset.m_items, sasrec_args)
        
    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        # Get embeddings from LightGCN
        # LightGCN returns (users, items). We only need items.
        # LightGCN items are 0-indexed (0 to M-1).
        # SASRec expects 0 to be padding, and items 1 to M.
        
        _, lightgcn_items = self.lightgcn.computer() # (M, D)
        
        # Add padding vector (zeros) at index 0
        padding = torch.zeros(1, lightgcn_items.shape[1], device=lightgcn_items.device)
        full_item_embeddings = torch.cat([padding, lightgcn_items], dim=0) # (M+1, D)
        
        return self.sasrec(user_ids, log_seqs, pos_seqs, neg_seqs, item_emb_matrix=full_item_embeddings)
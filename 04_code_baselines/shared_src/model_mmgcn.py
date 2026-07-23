import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MMGCN(nn.Module):
    def __init__(self, config, dataset, semantic_emb=None, content_emb=None):
        super(MMGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self.n_users = dataset.n_users
        self.n_items = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.n_layers = config['lightGCN_n_layers']
        self.device = config['device']
        
        # Load user/item graph
        self.Graph = dataset.getSparseGraph()
        
        # --- ID Embeddings (Interaction View) ---
        self.id_user_embedding = nn.Embedding(self.n_users, self.latent_dim)
        self.id_item_embedding = nn.Embedding(self.n_items, self.latent_dim)
        nn.init.xavier_uniform_(self.id_user_embedding.weight)
        nn.init.xavier_uniform_(self.id_item_embedding.weight)
        
        # --- Semantic View ---
        # User preference for semantic
        self.sem_user_embedding = nn.Embedding(self.n_users, self.latent_dim)
        nn.init.xavier_uniform_(self.sem_user_embedding.weight)
        
        if semantic_emb is not None:
            # Linear projection for semantic features (1024 -> 64)
            self.sem_item_trans = nn.Linear(semantic_emb.shape[1], self.latent_dim)
            self.sem_item_features = torch.tensor(semantic_emb, dtype=torch.float32, device=self.device)
        else:
            self.sem_item_embedding = nn.Embedding(self.n_items, self.latent_dim)
            nn.init.xavier_uniform_(self.sem_item_embedding.weight)
            self.sem_item_features = None

        # --- Content View ---
        # User preference for content
        self.con_user_embedding = nn.Embedding(self.n_users, self.latent_dim)
        nn.init.xavier_uniform_(self.con_user_embedding.weight)

        if content_emb is not None:
             # Linear projection for content features (1024 -> 64)
            self.con_item_trans = nn.Linear(content_emb.shape[1], self.latent_dim)
            self.con_item_features = torch.tensor(content_emb, dtype=torch.float32, device=self.device)
        else:
            self.con_item_embedding = nn.Embedding(self.n_items, self.latent_dim)
            nn.init.xavier_uniform_(self.con_item_embedding.weight)
            self.con_item_features = None
            
    def _get_item_embeddings(self):
        # ID View
        id_item = self.id_item_embedding.weight
        
        # Semantic View
        if self.sem_item_features is not None:
            sem_item = self.sem_item_trans(self.sem_item_features)
        else:
            sem_item = self.sem_item_embedding.weight
            
        # Content View
        if self.con_item_features is not None:
            con_item = self.con_item_trans(self.con_item_features)
        else:
            con_item = self.con_item_embedding.weight
            
        return id_item, sem_item, con_item

    def forward(self):
        # Get base embeddings
        id_item, sem_item, con_item = self._get_item_embeddings()
        
        # User embeddings for each view
        id_user = self.id_user_embedding.weight
        sem_user = self.sem_user_embedding.weight
        con_user = self.con_user_embedding.weight
        
        # Create all_emb list for stacking [id, sem, con]
        # Shape: [n_users + n_items, dim]
        ego_embeddings_id = torch.cat([id_user, id_item], dim=0)
        ego_embeddings_sem = torch.cat([sem_user, sem_item], dim=0)
        ego_embeddings_con = torch.cat([con_user, con_item], dim=0)
        
        all_embeddings_id = [ego_embeddings_id]
        all_embeddings_sem = [ego_embeddings_sem]
        all_embeddings_con = [ego_embeddings_con]
        
        # Graph Convolution
        # We use the same graph structure for all views (User-Item Interaction Graph)
        # In standard MMGCN practice, aggregation can happen per view
        
        if self.Graph.device != self.device:
            self.Graph = self.Graph.to(self.device)
            
        for k in range(self.n_layers):
            # Aggregation for each view
            # ID
            ego_embeddings_id = torch.sparse.mm(self.Graph, ego_embeddings_id)
            all_embeddings_id.append(ego_embeddings_id)
            
            # Semantic
            ego_embeddings_sem = torch.sparse.mm(self.Graph, ego_embeddings_sem)
            all_embeddings_sem.append(ego_embeddings_sem)
            
            # Content
            ego_embeddings_con = torch.sparse.mm(self.Graph, ego_embeddings_con)
            all_embeddings_con.append(ego_embeddings_con)
            
        # Mean Pooling over layers
        final_embeddings_id = torch.stack(all_embeddings_id, dim=1)
        final_embeddings_id = torch.mean(final_embeddings_id, dim=1)
        
        final_embeddings_sem = torch.stack(all_embeddings_sem, dim=1)
        final_embeddings_sem = torch.mean(final_embeddings_sem, dim=1)
        
        final_embeddings_con = torch.stack(all_embeddings_con, dim=1)
        final_embeddings_con = torch.mean(final_embeddings_con, dim=1)
        
        # Split into users and items
        user_id, item_id = torch.split(final_embeddings_id, [self.n_users, self.n_items])
        user_sem, item_sem = torch.split(final_embeddings_sem, [self.n_users, self.n_items])
        user_con, item_con = torch.split(final_embeddings_con, [self.n_users, self.n_items])
        
        # Fusion (Concatenation or Summation)
        # Here we use summation + concatenation or just summation based on standard MMGCN
        # Standard MMGCN usually sums or concatenates final representations
        # Let's use MEAN fusion for simplicity and stability compatible with LightGCN dims
        
        # We return the separate views AND the fused view if needed.
        # But BPR Loss usually computed on the FINAL representation.
        # MMGCN typically creates a final user/item representation.
        # h_u = h_u_id + h_u_sem + h_u_con
        # h_i = h_i_id + h_i_sem + h_i_con
        
        final_user = (user_id + user_sem + user_con) / 3
        final_item = (item_id + item_sem + item_con) / 3
        
        return final_user, final_item

    def getUsersRating(self, users):
        all_users, all_items = self.forward()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = torch.matmul(users_emb, items_emb.t())
        return rating
    
    def bpr_loss(self, users, pos, neg):
        all_users, all_items = self.forward()
        
        users_emb = all_users[users.long()]
        pos_emb = all_items[pos.long()]
        neg_emb = all_items[neg.long()]
        
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        
        loss = torch.mean(F.softplus(neg_scores - pos_scores))
        
        # Reg loss
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
                          
        return loss, reg_loss

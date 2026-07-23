"""
資料載入器 - 適用於 ShortVideo 資料集
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from time import time

class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getSparseGraph(self):
        raise NotImplementedError

class ShortVideoDataset(BasicDataset):
    """
    ShortVideo 資料集載入器 - 使用與 MicroLens 相同的格式
    """
    def __init__(self, path="../data/shortvideo"):
        print(f"Loading ShortVideo dataset from {path}")
        self.path = path
        self._trainDataSize = 0
        self._testDataSize = 0
        
        # 讀取 train.txt 和 test.txt
        train_file = os.path.join(path, 'train.txt')
        test_file = os.path.join(path, 'test.txt')
        
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Training file not found: {train_file}")
        
        # 載入訓練資料 (格式: user_id item_id)
        trainUser, trainItem = [], []
        self.n_user = 0
        self.m_item = 0
        
        print("Loading training data...")
        with open(train_file) as f:
            for line in f.readlines():
                if len(line) > 0:
                    line = line.strip().split(' ')
                    uid = int(line[0])
                    iid = int(line[1])
                    trainUser.append(uid)
                    trainItem.append(iid)
                    self.n_user = max(self.n_user, uid)
                    self.m_item = max(self.m_item, iid)
                    self._trainDataSize += 1
        
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)
        self.trainUniqueUsers = np.unique(self.trainUser)
        
        # 載入測試資料
        testUser, testItem = [], []
        print("Loading testing data...")
        if os.path.exists(test_file):
            with open(test_file) as f:
                for line in f.readlines():
                    if len(line) > 0:
                        line = line.strip().split(' ')
                        uid = int(line[0])
                        iid = int(line[1])
                        testUser.append(uid)
                        testItem.append(iid)
                        self.n_user = max(self.n_user, uid)
                        self.m_item = max(self.m_item, iid)
                        self._testDataSize += 1
        
        # 修正：因為 ID 從 0 開始，所以總數需要 +1
        self.n_user += 1  # max_user_id + 1 = 實際用戶數量
        self.m_item += 1  # max_item_id + 1 = 實際物品數量
        
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        
        self.Graph = None
        print(f"{self._trainDataSize:,} interactions for training")
        print(f"{self._testDataSize:,} interactions for testing")
        print(f"ShortVideo Sparsity: {(self._trainDataSize + self._testDataSize) / self.n_users / self.m_items:.6f}")
        
        # 建立使用者-物品互動矩陣
        print("Building user-item interaction matrix...")
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), 
                                     (self.trainUser, self.trainItem)),
                                    shape=(self.n_user, self.m_item))
        
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()  
        self.items_D[self.items_D == 0.] = 1.
        
        # 預先計算
        print("Pre-computing user positive items...")
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        print("Building test dictionary...")
        self.__testDict = self.__build_test()
        print("ShortVideo dataset is ready to go")
    
    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self._trainDataSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos
    
    def getUserItemFeedback(self, users, items):
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))
    
    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    
    def __build_test(self):
        """
        建立測試資料字典
        return: dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
    
    def getSparseGraph(self):
        """
        建立稀疏鄰接矩陣用於 GCN
        """
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, 
                                       self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)
            
            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce()
            print("adjacency matrix loaded")
        return self.Graph
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
    
    def __getitem__(self, index):
        user = self.trainUniqueUsers[index]
        return user
    
    def __len__(self):
        return len(self.trainUniqueUsers)

# 為了相容性，也可以使用之前的 MicroLensDataset 類別
class MicroLensDataset(BasicDataset):
    """
    MicroLens-100K 資料集載入器
    """
    def __init__(self, path="../data/lightgcn"):
        print(f"Loading MicroLens dataset from {path}")
        self.path = path
        self._trainDataSize = 0
        self._testDataSize = 0
        
        # 讀取 train.txt 和 test.txt
        train_file = os.path.join(path, 'train.txt')
        test_file = os.path.join(path, 'test.txt')
        
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Training file not found: {train_file}")
        
        # 載入訓練資料 (每行格式: user_id item_id)
        trainUniqueUsers, trainUser, trainItem = [], [], []
        unique_users = set()
        self.n_user = 0
        self.m_item = 0
        
        with open(train_file) as f:
            for line in f.readlines():
                if len(line) > 0:
                    line = line.strip().split(' ')
                    uid = int(line[0])
                    item = int(line[1])
                    unique_users.add(uid)
                    trainUser.append(uid)
                    trainItem.append(item)
                    self.m_item = max(self.m_item, item)
                    self.n_user = max(self.n_user, uid)
                    self._trainDataSize += 1
        
        trainUniqueUsers = list(unique_users)
        
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)
        
        # 載入測試資料 (每行格式: user_id item_id)
        testUniqueUsers, testUser, testItem = [], [], []
        test_unique_users = set()
        if os.path.exists(test_file):
            with open(test_file) as f:
                for line in f.readlines():
                    if len(line) > 0:
                        line = line.strip().split(' ')
                        uid = int(line[0])
                        item = int(line[1])
                        test_unique_users.add(uid)
                        testUser.append(uid)
                        testItem.append(item)
                        self.m_item = max(self.m_item, item)
                        self.n_user = max(self.n_user, uid)
                        self._testDataSize += 1
        
        testUniqueUsers = list(test_unique_users)
        
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        
        # 維度需要+1因為索引從0開始
        self.n_user += 1
        self.m_item += 1
        
        self.Graph = None
        print(f"{self._trainDataSize} interactions for training")
        print(f"{self._testDataSize} interactions for testing")
        print(f"MicroLens Sparsity: {(self._trainDataSize + self._testDataSize) / self.n_user / self.m_item:.6f}")
        
        # 建立使用者-物品互動矩陣
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), 
                                     (self.trainUser, self.trainItem)),
                                    shape=(self.n_user, self.m_item))
        
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()  
        self.items_D[self.items_D == 0.] = 1.
        
        # 預先計算
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        print("MicroLens dataset is ready to go")
    
    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self._trainDataSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos
    
    def getUserItemFeedback(self, users, items):
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))
    
    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    
    def __build_test(self):
        """
        建立測試資料字典
        return: dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
    
    def getSparseGraph(self):
        """
        建立稀疏鄰接矩陣用於 GCN
        """
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, 
                                       self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)
            
            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce()
            print("adjacency matrix loaded")
        return self.Graph
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
    
    def __getitem__(self, index):
        user = self.trainUniqueUsers[index]
        return user
    
    def __len__(self):
        return len(self.trainUniqueUsers)
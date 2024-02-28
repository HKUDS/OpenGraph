import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from params import args
import scipy.sparse as sp
from Utils.TimeLogger import log
import torch as t
import torch.utils.data as data
import torch_geometric.transforms as T
from model import InitialProjector
import os

class MultiDataHandler:
    def __init__(self, trn_datasets, tst_datasets):
        all_datasets = list(set(trn_datasets + tst_datasets))
        self.trn_handlers = []
        self.tst_handlers = []
        for data_name in all_datasets:
            trn_flag = data_name in trn_datasets
            tst_flag = data_name in tst_datasets
            handler = DataHandler(data_name, trn_flag, tst_flag)
            if trn_flag:
                self.trn_handlers.append(handler)
            if tst_flag:
                self.tst_handlers.append(handler)
        self.make_joint_trn_loader()

    def make_joint_trn_loader(self):
        trn_data = TrnData(self.trn_handlers)
        self.trn_loader = data.DataLoader(trn_data, batch_size=1, shuffle=True, num_workers=0)
    
    def remake_initial_projections(self):
        for i in range(len(self.trn_handlers)):
            self.remake_one_initial_projection(i)
    
    def remake_one_initial_projection(self, idx):
        trn_handler = self.trn_handlers[idx]
        trn_handler.initial_projector = InitialProjector(trn_handler.asym_adj)

class DataHandler:
    def __init__(self, data_name, trn_flag, tst_flag):
        self.data_name = data_name
        self.trn_flag = trn_flag
        self.tst_flag = tst_flag
        self.get_data_files()
        self.load_data()
    
    def get_data_files(self):
        predir = os.path.join(args.data_dir, self.data_name)
        self.trnfile = os.path.join(predir, 'trn_mat.pkl')
        self.tstfile = os.path.join(predir, 'tst_mat.pkl')
        self.valfile = os.path.join(predir, 'val_mat.pkl')
        if not os.path.exists(self.valfile):
            self.valfile = self.tstfile

    def load_one_file(self, filename):
        with open(filename, 'rb') as fs:
            ret = (pickle.load(fs)).astype(np.float32)
        if type(ret) != coo_matrix:
            ret = sp.coo_matrix(ret)
        return ret

    def normalize_adj(self, mat):
        degree = np.array(mat.sum(axis=-1))
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        if mat.shape[0] == mat.shape[1]:
            return mat.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()
        else:
            tem = d_inv_sqrt_mat.dot(mat)
            col_degree = np.array(mat.sum(axis=0))
            d_inv_sqrt = np.reshape(np.power(col_degree, -0.5), [-1])
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
            d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
            return tem.dot(d_inv_sqrt_mat).tocoo()

    def unique_numpy(self, row, col):
        hash_vals = row * self.node_num + col
        hash_vals = np.unique(hash_vals).astype(np.int64)
        col = hash_vals % self.node_num
        row = (hash_vals - col).astype(np.int64) // self.node_num
        return row, col
    
    def make_torch_adj(self, mat):
        if mat.shape[0] == mat.shape[1]:
            # to symmetric
            if self.data_name in ['ddi']:
                _row = mat.row
                _col = mat.col
                row = np.concatenate([_row, _col]).astype(np.int64)
                col = np.concatenate([_col, _row]).astype(np.int64)
                # row, col = self.unique_numpy(row, col)
                data = mat.data
                data = np.concatenate([data, data]).astype(np.float32)
            else:
                row, col = mat.row, mat.col
                data = mat.data
            # data = np.ones_like(row)
            mat = coo_matrix((data, (row, col)), mat.shape)
            if args.selfloop == 1:
                mat = (mat + sp.eye(mat.shape[0])) * 1.0
        normed_asym_mat = self.normalize_adj(mat)
        row = t.from_numpy(normed_asym_mat.row).long()
        col = t.from_numpy(normed_asym_mat.col).long()
        idxs = t.stack([row, col], dim=0)
        vals = t.from_numpy(normed_asym_mat.data).float()
        shape = t.Size(normed_asym_mat.shape)
        asym_adj = t.sparse.FloatTensor(idxs, vals, shape)
        if mat.shape[0] == mat.shape[1]:
            return asym_adj, asym_adj
        else:
            # make ui adj
            a = sp.csr_matrix((self.user_num, self.user_num))
            b = sp.csr_matrix((self.item_num, self.item_num))
            mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
            mat = (mat != 0) * 1.0
            if args.selfloop == 1:
                mat = (mat + sp.eye(mat.shape[0])) * 1.0
            mat = self.normalize_adj(mat)

            # make cuda tensor
            idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
            vals = t.from_numpy(mat.data.astype(np.float32))
            shape = t.Size(mat.shape)
            return t.sparse.FloatTensor(idxs, vals, shape), asym_adj

    def load_data(self):
        trn_mat = self.load_one_file(self.trnfile)
        # if self.trn_flag:
        self.trn_mat = trn_mat
        if trn_mat.shape[0] != trn_mat.shape[1]:
            self.user_num, self.item_num = trn_mat.shape
            self.node_num = self.user_num + self.item_num
            print('Dataset: {data_name}, User num: {user_num}, Item num: {item_num}, Node num: {node_num}, Edge num: {edge_num}'.format(data_name=self.data_name, user_num=self.user_num, item_num=self.item_num, node_num=self.node_num, edge_num=trn_mat.nnz))
        else:
            self.node_num = trn_mat.shape[0]
            print('Dataset: {data_name}, Node num: {node_num}, Edge num: {edge_num}'.format(data_name=self.data_name, node_num=self.node_num, edge_num=trn_mat.nnz))
        self.torch_adj, self.asym_adj = self.make_torch_adj(trn_mat)
        if args.cache_proj:
            self.asym_adj = self.asym_adj.to(args.devices[0])
        if args.cache_adj:
            self.torch_adj = self.torch_adj.to(args.devices[0])

        self.initial_projector = InitialProjector(self.asym_adj)
        
        if self.tst_flag:
            val_mat = self.load_one_file(self.valfile)
            val_data = TstData(val_mat, trn_mat)
            self.val_loader = data.DataLoader(val_data, batch_size=args.tst_batch, shuffle=False, num_workers=0)
            tst_mat = self.load_one_file(self.tstfile)
            tst_data = TstData(tst_mat, trn_mat)
            self.tst_loader = data.DataLoader(tst_data, batch_size=args.tst_batch, shuffle=False, num_workers=0)

class TstData(data.Dataset):
    def __init__(self, coomat, trn_mat):
        self.csrmat = (trn_mat.tocsr() != 0) * 1.0

        tstLocs = [None] * coomat.shape[0]
        tst_nodes = set()
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            if tstLocs[row] is None:
                tstLocs[row] = list()
            tstLocs[row].append(col)
            tst_nodes.add(row)
        tst_nodes = np.array(list(tst_nodes))
        self.tst_nodes = tst_nodes
        self.tstLocs = tstLocs

    def __len__(self):
        return len(self.tst_nodes)

    def __getitem__(self, idx):
        return self.tst_nodes[idx]

class TrnData(data.Dataset):
    def __init__(self, trn_handlers):
        self.dataset_num = len(trn_handlers)
        self.trn_handlers = trn_handlers
        self.ancs_list = [None] * self.dataset_num
        self.poss_list = [None] * self.dataset_num
        self.negs_list = [None] * self.dataset_num
        self.edge_nums = [None] * self.dataset_num
        self.sample_nums = [None] * self.dataset_num
        for i, handler in enumerate(self.trn_handlers):
            trn_mat = handler.trn_mat
            ancs = np.array(trn_mat.row)
            poss = np.array(trn_mat.col)
            self.ancs_list[i] = ancs
            self.poss_list[i] = poss
            self.edge_nums[i] = len(ancs)
            self.sample_nums[i] = self.edge_nums[i] // args.batch + (1 if self.edge_nums[i] % args.batch != 0 else 0)
        self.total_sample_num = sum(self.sample_nums)
        self.samples = [None] * self.total_sample_num
    
    def data_shuffling(self):
        sample_idx = 0
        for i in range(self.dataset_num):
            edge_num = self.edge_nums[i]
            perms = np.random.permutation(edge_num)
            handler = self.trn_handlers[i]
            asym_flag = handler.trn_mat.shape[0] != handler.trn_mat.shape[1]
            cand_num = handler.item_num if asym_flag else handler.node_num
            self.negs_list[i] = self.neg_sampling(self.ancs_list[i], handler.trn_mat.todok(), cand_num)
            # self.negs_list[i] = np.random.randint(cand_num, size=edge_num)
            for j in range(self.sample_nums[i]):
                st_idx = j * args.batch
                ed_idx = min((j + 1) * args.batch, edge_num)
                pick_idxs = perms[st_idx: ed_idx]
                ancs = self.ancs_list[i][pick_idxs]
                poss = self.poss_list[i][pick_idxs]
                negs = self.negs_list[i][pick_idxs]
                if asym_flag:
                    poss += handler.user_num
                    negs += handler.user_num
                self.samples[sample_idx] = (ancs, poss, negs, i)
                sample_idx += 1
        assert sample_idx == self.total_sample_num
    
    def neg_sampling(self, ancs, dokmat, cand_num):
        negs = np.zeros_like(ancs)
        for i in range(len(ancs)):
            u = ancs[i]
            while True:
                i_neg = np.random.randint(cand_num)
                if (u, i_neg) not in dokmat:
                    break
            negs[i] = i_neg
        return negs
    
    def __len__(self):
        return self.total_sample_num
    
    def __getitem__(self, idx):
        ancs, poss, negs, adj_id = self.samples[idx]
        return ancs, poss, negs, adj_id

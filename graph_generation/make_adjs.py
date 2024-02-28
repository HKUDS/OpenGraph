import pickle
import os
import argparse
from scipy.sparse import coo_matrix
import numpy as np
import random
import networkx as nx

descs = {
    'data_name': 'gen_data_ecommerce',
}
params = {
    'itmfusion': True,
    'kcore': 0,
    'sep': 1,
    'min_base': 0,
    'max_base': 100,
}
parser = argparse.ArgumentParser(description='Dataset information')
parser.add_argument('--gen_iter', default=0, type=int, help='maximum generation iteration')
args = parser.parse_args()



file_root = 'gen_results/datasets/{data_name}/'.format(data_name=descs['data_name'])
fuse_file_path = file_root + 'res/interaction_fuse_iter-{iter}.pkl'.format(iter=args.gen_iter)

def get_all_bases():
    bases = set()
    prefix = 'interaction_base-'
    suffix = '_iter-'
    for filename in os.listdir(file_root):
        if prefix in filename and suffix in filename:
            prefix_idx = len(prefix)
            suffix_idx = filename.index(suffix)
            base = int(filename[prefix_idx: suffix_idx])
            if base >= params['min_base'] and base <= params['max_base']:
                bases.add(base)
    bases = list(bases)
    bases.sort()
    return bases

def fuse_bases():
    if os.path.exists(fuse_file_path):
        print('Fused interaction file exists! REUSING!')
        print('This may happen inadverdently!')
        exit()
        with open(fuse_file_path, 'rb') as fs:
            interactions = pickle.load(fs)
            return interactions
    all_bases = get_all_bases()
    interactions = []
    for gen_base in all_bases:
        file_path = None
        for iter in range(args.gen_iter, -1, -1):
            file_path = file_root + 'interaction_base-{gen_base}_iter-{gen_iter}.pkl'.format(gen_base=gen_base, gen_iter=iter)
            if os.path.exists(file_path):
                break
        with open(file_path, 'rb') as fs:
            tem_cur_base_interactions = pickle.load(fs)
            cur_base_interactions = []
            for i in range(len(tem_cur_base_interactions)):
                cur_base_interactions.append(tem_cur_base_interactions[i])
        interactions += cur_base_interactions
    
    new_interactions = []
    for i in range(len(interactions)):
        if i % params['sep'] == 0:
            # interactions[i] = interactions[i][:len(interactions[i]) // 3]
            new_interactions.append(interactions[i])
    interactions = new_interactions
    # interactions = new_interactions[:20000]

    with open(fuse_file_path, 'wb+') as fs:
        pickle.dump(interactions, fs)
    return interactions

def make_id_map(interactions, itm_criteria=None):
    u_num = len(interactions)
    i_set = set()
    i_cnt = dict()
    for interaction in interactions:
        for item in interaction:
            num_idx = item.index(' #')
            tem_item = item if not params['itmfusion'] else item[:num_idx]
            i_set.add(tem_item)
            if tem_item not in i_cnt:
                i_cnt[tem_item] = 0
            i_cnt[tem_item] += 1
    i_list = list(i_set)
    if itm_criteria is not None:
        tem_i_list = list()
        for item in i_list:
            if itm_criteria(i_cnt[item]):
                tem_i_list.append(item)
        print('Filtering {new_num} / {old_num}'.format(new_num=len(tem_i_list), old_num=len(i_list)))
        i_list = tem_i_list
    i_num = len(i_list)
    i_mapp = dict()
    for i, item in enumerate(i_list):
        i_mapp[item] = i
    rows = []
    cols = []
    for uid, interaction in enumerate(interactions):
        for item in interaction:
            num_idx = item.index(' #')
            tem_item = item if not params['itmfusion'] else item[:num_idx]
            if tem_item not in i_mapp:
                continue
            iid = i_mapp[tem_item]
            rows.append(uid)
            cols.append(iid)
    return rows, cols, i_mapp, u_num, i_num

def id_map(nodes):
    uniq_nodes = list(set(nodes))
    dic = dict()
    for i, node in enumerate(uniq_nodes):
        dic[node] = i
    return dic

def k_core(rows, cols, i_mapp, i_num, k):
    edge_list = list(map(lambda idx: (rows[idx] + i_num, cols[idx]), range(len(rows))))
    G = nx.Graph(edge_list)
    edge_list = list(nx.k_core(G, k=k).edges())
    rows = [None] * len(edge_list)
    cols = [None] * len(edge_list)
    for i, edge in enumerate(edge_list):
        if edge[0] < i_num:
            rows[i] = edge[1] - i_num
            cols[i] = edge[0]
        else:
            rows[i] = edge[0] - i_num
            cols[i] = edge[1]
    row_map = id_map(rows)
    col_map = id_map(cols)
    new_rows = list(map(lambda x: row_map[x], rows))
    new_cols = list(map(lambda x: col_map[x], cols))
    new_i_mapp = dict()
    for key in i_mapp:
        tem_item = i_mapp[key]
        if tem_item not in col_map:
            continue
        new_i_mapp[key] = col_map[tem_item]
    return new_rows, new_cols, new_i_mapp, len(row_map), len(col_map)

def make_mat(rows, cols, st, ed, u_num, i_num, perm, decrease=False):
    rows = np.array(rows)[perm]
    cols = np.array(cols)[perm]
    rows = rows[st: ed]
    cols = cols[st: ed]
    if decrease:
        rows = rows[:len(rows)//3]
        cols = cols[:len(cols)//3]
    vals = np.ones_like(rows)
    return coo_matrix((vals, (rows, cols)), shape=[u_num, i_num])

def data_split(rows, cols, u_num, i_num):
    leng = len(rows)
    perm = np.random.permutation(leng)
    trn_split = int(leng * 0.7)
    val_split = int(leng * 0.75)
    trn_mat = make_mat(rows, cols, 0, trn_split, u_num, i_num, perm)
    val_mat = make_mat(rows, cols, trn_split, val_split, u_num, i_num, perm)
    tst_mat = make_mat(rows, cols, val_split, leng, u_num, i_num, perm, decrease=True)
    return trn_mat, val_mat, tst_mat

interactions = fuse_bases()
rows, cols, i_mapp, u_num, i_num = make_id_map(interactions)#, lambda x: x>20)
if params['kcore'] != 1:
    rows, cols, i_mapp, u_num, i_num = k_core(rows, cols, i_mapp, i_num, params['kcore'])
print('U NUM', u_num, 'I Num', i_num, 'E Num', len(rows))
with open(file_root + 'res/iter-{gen_iter}_imap.pkl'.format(gen_iter=args.gen_iter), 'wb+') as fs:
    pickle.dump(i_mapp, fs)
trn_mat, val_mat, tst_mat = data_split(rows, cols, u_num, i_num)
with open(file_root + 'res/iter-{gen_iter}_train.pkl'.format(gen_iter=args.gen_iter), 'wb+') as fs:
    pickle.dump(trn_mat, fs)
with open(file_root + 'res/iter-{gen_iter}_valid.pkl'.format(gen_iter=args.gen_iter), 'wb+') as fs:
    pickle.dump(val_mat, fs)
with open(file_root + 'res/iter-{gen_iter}_test.pkl'.format(gen_iter=args.gen_iter), 'wb+') as fs:
    pickle.dump(tst_mat, fs)
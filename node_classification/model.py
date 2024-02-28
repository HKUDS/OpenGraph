import torch as t
from torch import nn
import torch.nn.functional as F
from params import args
import numpy as np
from Utils.TimeLogger import log
from torch.nn import MultiheadAttention
from time import time

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform_

class InitialProjector(nn.Module):
    def __init__(self, adj, input_is_embeds=False):
        super(InitialProjector, self).__init__()

        if input_is_embeds:
            projection = adj
            if args.cache_proj:
                projection = projection.to(args.devices[0])
            else:
                projection = projection.cpu()
            self.proj_embeds = nn.Parameter(projection)
            t.cuda.empty_cache()
            return
        if args.proj_method == 'uniform':
            self.proj_embeds = nn.Parameter(self.uniform_proj(adj))
        elif args.proj_method == 'lowrank_uniform':
            self.proj_embeds = nn.Parameter(self.lowrank_uniform_proj(adj))
        elif args.proj_method == 'svd':
            self.proj_embeds = nn.Parameter(self.svd_proj(adj))
        elif args.proj_method == 'both':
            self.proj_embeds = nn.Parameter(self.uniform_proj(adj) + self.svd_proj(adj))
        elif args.proj_method == 'id':
            self.proj_embeds = nn.Parameter(self.id_proj(adj))
        else:
            raise Exception('Unrecognized Initial Embedding')
        t.cuda.empty_cache()
    
    def uniform_proj(self, adj):
        node_num = adj.shape[0] if adj.shape[0] == adj.shape[1] else adj.shape[0] + adj.shape[1]
        projection = init(t.empty(node_num, args.latdim))
        if args.cache_proj:
            projection = projection.to(args.devices[0])
        return projection
    
    def id_proj(self, adj):
        node_num = adj.shape[0] if adj.shape[0] == adj.shape[1] else adj.shape[0] + adj.shape[1]
        return t.eye(node_num)

    def lowrank_uniform_proj(self, adj):
        node_num = adj.shape[0] + adj.shape[1]
        rank = 16
        projection1 = init(t.empty(node_num, rank))
        projection2 = init(t.empty(rank, args.latdim))
        projection = projection1 @ projection2
        if args.cache_proj:
            projection = projection.to(args.devices[0])
        return projection
    
    def svd_proj(self, adj):
        if not args.cache_proj:
            adj = adj.to(args.devices[0])
        q = args.latdim
        if args.latdim > adj.shape[0] or args.latdim > adj.shape[1]:
            dim = min(adj.shape[0], adj.shape[1])
            svd_u, s, svd_v = t.svd_lowrank(adj, q=dim, niter=args.niter)
            svd_u = t.concat([svd_u, t.zeros([svd_u.shape[0], args.latdim-dim]).to(args.devices[0])], dim=1)
            svd_v = t.concat([svd_v, t.zeros([svd_v.shape[0], args.latdim-dim]).to(args.devices[0])], dim=1)
            s = t.concat([s, t.zeros(args.latdim-dim).to(args.devices[0])])
        else:
            svd_u, s, svd_v = t.svd_lowrank(adj, q=q, niter=args.niter)
        svd_u = svd_u @ t.diag(t.sqrt(s))
        svd_v = svd_v @ t.diag(t.sqrt(s))
        if adj.shape[0] != adj.shape[1]:
            projection = t.concat([svd_u, svd_v], dim=0)
        else:
            projection = svd_u + svd_v
        if not args.cache_proj:
            projection = projection.cpu()
        return projection

    def forward(self):
        return ((self.proj_embeds))#.cuda()#[perms, :]

class TopoEncoder(nn.Module):
    def __init__(self):
        super(TopoEncoder, self).__init__()

        self.layer_norm = nn.LayerNorm(args.latdim, elementwise_affine=False)#, dtype=t.bfloat16)
    
    def forward(self, adj, embeds):
        embeds = self.layer_norm(embeds)
        embeds_list = []
        if args.gnn_layer == 0:
            embeds_list.append(embeds)
        for i in range(args.gnn_layer):
            embeds = t.spmm(adj, embeds)
            embeds_list.append(embeds)
        embeds = sum(embeds_list)
        # embeds = t.concat([embeds_list[-1][:user_num], embeds_list[-2][user_num:]], dim=0)
        embeds = embeds#.to(t.bfloat16)
        return embeds
    
class GraphTransformer(nn.Module):
    def __init__(self):
        super(GraphTransformer, self).__init__()
        self.gt_layers = nn.Sequential(*[GTLayer() for i in range(args.gt_layer)])

    def forward(self, embeds):
        for i, layer in enumerate(self.gt_layers):
            embeds = layer(embeds) / args.scale_layer
        return embeds

class GTLayer(nn.Module):
    def __init__(self):
        super(GTLayer, self).__init__()
        self.multi_head_attention = MultiheadAttention(args.latdim, args.head, dropout=0.1, bias=False)#, dtype=t.bfloat16)
        self.dense_layers = nn.Sequential(*[FeedForwardLayer(args.latdim, args.latdim, bias=True, act=args.act) for _ in range(2)])# bias=False
        self.layer_norm1 = nn.LayerNorm(args.latdim, elementwise_affine=True)#, dtype=t.bfloat16)
        self.layer_norm2 = nn.LayerNorm(args.latdim, elementwise_affine=True)#, dtype=t.bfloat16)
        self.fc_dropout = nn.Dropout(p=args.drop_rate)

    def _attention(self, anchor_embeds, embeds):
        q_embeds = t.einsum('ne,ehd->nhd', anchor_embeds, self.Q)
        k_embeds = t.einsum('ne,ehd->nhd', embeds, self.K)
        v_embeds = t.einsum('ne,ehd->nhd', embeds, self.V)
        att = t.einsum('khd,nhd->knh', q_embeds, k_embeds) / np.sqrt(args.latdim / args.head)
        att = t.softmax(att, dim=1)
        res = t.einsum('knh,nhd->khd', att, v_embeds).reshape([-1, args.latdim])
        res = self.att_linear(res)
        return res
    
    def _pick_anchors(self, embeds):
        perm = t.randperm(embeds.shape[0])
        anchors = perm[:args.anchor]
        return embeds[anchors]
    
    def print_nodewise_std(self, embeds):
        mean = embeds.mean(0)
        std = (embeds - mean).square().mean(0).sqrt().mean()
        print(embeds)
        print(std.item())
   
    def forward(self, embeds):
        anchor_embeds = self._pick_anchors(embeds)
        _anchor_embeds, _ = self.multi_head_attention(anchor_embeds, embeds, embeds)
        anchor_embeds = _anchor_embeds + anchor_embeds
        _embeds, _ = self.multi_head_attention(embeds, anchor_embeds, anchor_embeds, need_weights=False)
        embeds = self.layer_norm1(_embeds + embeds)
        _embeds = self.fc_dropout(self.dense_layers(embeds))
        embeds = (self.layer_norm2(_embeds + embeds))
        return embeds

class FeedForwardLayer(nn.Module):
    def __init__(self, in_feat, out_feat, bias=True, act=None):
        super(FeedForwardLayer, self).__init__()
        self.linear = nn.Linear(in_feat, out_feat, bias=bias)#, dtype=t.bfloat16)
        if act == 'identity' or act is None:
            self.act = None
        elif act == 'leaky':
            self.act = nn.LeakyReLU(negative_slope=args.leaky)
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'relu6':
            self.act = nn.ReLU6()
        else:
            raise Exception('Error')
    
    def forward(self, embeds):
        if self.act is None:
            return self.linear(embeds)
        return self.act(self.linear(embeds))

class Masker(nn.Module):
    def __init__(self):
        super(Masker, self).__init__()

    def forward(self, adj, edges):
        if args.mask_method is None or args.mask_method == 'none':
            return adj
        elif args.mask_method == 'trn':
            node_num = adj.shape[0] + adj.shape[1]
            rows = adj._indices()[0, :]
            cols = adj._indices()[1, :]
            pck_rows, pck_cols = edges

            hashvals = rows * node_num + cols
            pck_hashvals1 = pck_rows * node_num + pck_cols
            pck_hashvals2 = pck_cols * node_num + pck_rows
            pck_hashvals = t.concat([pck_hashvals1, pck_hashvals2])

            if args.mask_alg == 'cross':
                masked_hashvals = self._mask_by_cross(hashvals, pck_hashvals)
            elif args.mask_alg == 'linear':
                masked_hashvals = self._mask_by_linear(hashvals, pck_hashvals)

            cols = masked_hashvals % node_num
            rows = t.div((masked_hashvals - cols).long(), node_num, rounding_mode='trunc').long()

            adj = t.sparse.FloatTensor(t.stack([rows, cols], dim=0), t.ones_like(rows, dtype=t.float32).to(args.devices[0]), adj.shape)
            return self._normalize_adj(adj)
        elif args.mask_method == 'random':
            return self._random_mask_edge(adj)
    
    def _mask_by_cross(self, hashvals, pck_hashvals):
        for i in range(args.batch * 2 // args.mask_bat):
            bat_pck_hashvals = pck_hashvals[i * args.mask_bat: (i+1) * args.mask_bat]
            idct = (hashvals.view([-1, 1]) - bat_pck_hashvals.view([1, -1]) == 0).sum(-1).bool()
            hashvals = hashvals[t.logical_not(idct)]
        return hashvals
    
    def _mask_by_linear(self, hashvals, pck_hashvals):
        hashvals = t.unique(hashvals)
        pck_hashvals = t.unique(pck_hashvals)
        hashvals = t.concat([hashvals, pck_hashvals])
        hashvals, counts = t.unique(hashvals, return_counts=True)
        hashvals = hashvals[counts==1]
        return hashvals

    def _random_mask_edge(self, adj):
        if args.random_mask_rate == 0.0:
            return adj
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = ((t.rand(edgeNum) + 1.0 - args.random_mask_rate).floor()).type(t.bool)
        newIdxs = idxs[:, mask]
        newVals = t.ones(newIdxs.shape[1]).to(args.devices[0]).float()
        return self._normalize_adj(t.sparse.FloatTensor(newIdxs, newVals, adj.shape))
    
    def _normalize_adj(self, adj):
        row_degree = t.pow(t.sparse.sum(adj, dim=1).to_dense(), 0.5)
        col_degree = t.pow(t.sparse.sum(adj, dim=0).to_dense(), 0.5)
        newRows, newCols = adj._indices()[0, :], adj._indices()[1, :]
        rowNorm, colNorm = row_degree[newRows], col_degree[newCols]
        newVals = adj._values() / rowNorm / colNorm
        return t.sparse.FloatTensor(adj._indices(), newVals, adj.shape)

class OpenGraph(nn.Module):
    def __init__(self):
        super(OpenGraph, self).__init__()
        self.topoEncoder = TopoEncoder().to(args.devices[0])
        self.graphTransformer = GraphTransformer().to(args.devices[1])
        self.masker = Masker().to(args.devices[0])
    
    def forward(self, adj, initial_projector, user_num):
        topo_embeds = self.topoEncoder(adj, initial_projector(), user_num).to(args.devices[1])
        final_embeds = self.graphTransformer(topo_embeds)
        return final_embeds

    def pred_norm(self, pos_preds, neg_preds):
        pos_preds_num = pos_preds.shape[0]
        neg_preds_shape = neg_preds.shape
        preds = t.concat([pos_preds, neg_preds.view(-1)])
        preds = preds - preds.max()
        pos_preds = preds[:pos_preds_num]
        neg_preds = preds[pos_preds_num:].view(neg_preds_shape)
        return pos_preds, neg_preds
    
    def cal_loss(self, batch_data, adj, initial_projector):
        ancs, poss, negs = batch_data
        with t.no_grad():
            masked_adj = self.masker(adj, (ancs.to(args.devices[0]), (poss.to(args.devices[0]))))
            initial_embeds = initial_projector()
            topo_embeds = self.topoEncoder(masked_adj, initial_embeds).to(args.devices[1])
        ancs, poss, negs = ancs.to(args.devices[1]), poss.to(args.devices[1]), negs.to(args.devices[1])
        input_seq = t.concat([ancs, poss, negs])
        input_seq = topo_embeds[input_seq]
        final_embeds = self.graphTransformer(input_seq)
        anc_embeds, pos_embeds, neg_embeds = t.split(final_embeds[:ancs.shape[0] * 3], [ancs.shape[0]] * 3)
        # anc_embeds, pos_embeds, neg_embeds = final_embeds[ancs], final_embeds[poss], final_embeds[negs]
        if final_embeds.isinf().any() or final_embeds.isnan().any():
            raise Exception('Final embedding fails')

        pos_preds, neg_preds = self.pred_norm((anc_embeds * pos_embeds).sum(-1), anc_embeds @ neg_embeds.T)
        if pos_preds.isinf().any() or pos_preds.isnan().any() or neg_preds.isinf().any() or neg_preds.isnan().any():
            raise Exception('Preds fails')
        pos_loss = pos_preds
        neg_loss = (neg_preds.exp().sum(-1) + pos_preds.exp() + 1e-8).log()
        pre_loss = -(pos_loss - neg_loss).mean()
        
        if t.isinf(pre_loss).any() or t.isnan(pre_loss).any():
            raise Exception('NaN or Inf')

        reg_loss = sum(list(map(lambda W: W.norm(2).square() * args.reg, self.parameters())))
        loss_dict = {'preloss': pre_loss, 'regloss': reg_loss, 'posloss': pos_loss.mean(), 'negloss': neg_loss.mean()}
        return pre_loss + reg_loss, loss_dict

    def cal_loss_node(self, batch_data, adj, initial_projector):
        ancs, labels = batch_data
        poss = labels + adj.shape[0] - args.class_num
        negs = t.from_numpy(np.array(list(range(args.class_num)))).to(t.int64).cuda() + adj.shape[0] - args.class_num
        with t.no_grad():
            masked_adj = self.masker(adj, (ancs.to(args.devices[0]), (poss.to(args.devices[0]))))
            initial_embeds = initial_projector()
            topo_embeds = self.topoEncoder(masked_adj, initial_embeds).to(args.devices[1])
        ancs, poss, negs = ancs.to(args.devices[1]), poss.to(args.devices[1]), negs.to(args.devices[1])
        input_seq = t.concat([ancs, poss, negs])
        input_seq = topo_embeds[input_seq]
        final_embeds = self.graphTransformer(input_seq)
        # anc_embeds, pos_embeds, neg_embeds = t.split(final_embeds[:ancs.shape[0] * 3], [ancs.shape[0]] * 3)
        anc_embeds = final_embeds[:ancs.shape[0]]
        pos_embeds = final_embeds[ancs.shape[0]:ancs.shape[0]+poss.shape[0]]
        neg_embeds = final_embeds[-negs.shape[0]:]
        # anc_embeds, pos_embeds, neg_embeds = final_embeds[ancs], final_embeds[poss], final_embeds[negs]
        if final_embeds.isinf().any() or final_embeds.isnan().any():
            raise Exception('Final embedding fails')

        pos_preds, neg_preds = self.pred_norm((anc_embeds * pos_embeds).sum(-1), anc_embeds @ neg_embeds.T)
        if pos_preds.isinf().any() or pos_preds.isnan().any() or neg_preds.isinf().any() or neg_preds.isnan().any():
            raise Exception('Preds fails')
        pos_loss = pos_preds
        neg_loss = (neg_preds.exp().sum(-1) + pos_preds.exp() + 1e-8).log()
        pre_loss = -(pos_loss - neg_loss).mean()
        
        if t.isinf(pre_loss).any() or t.isnan(pre_loss).any():
            raise Exception('NaN or Inf')

        reg_loss = sum(list(map(lambda W: W.norm(2).square() * args.reg, self.parameters())))
        preds = anc_embeds @ neg_embeds.T
        loss_dict = {'preloss': pre_loss, 'regloss': reg_loss, 'posloss': pos_loss.mean(), 'negloss': neg_loss.mean(), 'preds': t.argmax(preds, dim=-1)}
        return pre_loss + reg_loss, loss_dict   
    
    def pred_for_node_test(self, nodes, adj, initial_projector, rerun_embed=True):
        if rerun_embed:
            final_embeds = self.graphTransformer(self.topoEncoder(adj, initial_projector()).to(args.devices[1]))
            self.final_embeds = final_embeds
        final_embeds = self.final_embeds
        pck_embeds = final_embeds[nodes]
        class_embeds = final_embeds[-args.class_num:]
        preds = pck_embeds @ class_embeds.T
        return t.argmax(preds, dim=-1)

class ALRS:
    def __init__(self, optimizer, loss_threshold=0.01, loss_ratio_threshold=0.01, decay_rate=0.97):
        self.optimizer = optimizer
        self.loss_threshold = loss_threshold
        self.decay_rate = decay_rate
        self.loss_ratio_threshold = loss_ratio_threshold
        self.last_loss = 1e9
    
    def step(self, loss):
        delta = self.last_loss - loss
        if delta < self.loss_threshold and delta / self.last_loss < self.loss_ratio_threshold:
            for group in self.optimizer.param_groups:
                group['lr'] *= self.decay_rate
        self.last_loss = loss

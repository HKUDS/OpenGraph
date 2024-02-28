import torch as t
from torch import nn
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from params import args
from model import OpenGraph, ALRS
from data_handler import DataHandler, MultiDataHandler
import numpy as np
import pickle
import os
import setproctitle
import time

class Exp:
    def __init__(self, multi_handler):
        self.multi_handler = multi_handler
        self.metrics = dict()
        trn_mets = ['Loss', 'preLoss']
        tst_mets = ['Recall', 'NDCG']
        mets = trn_mets + tst_mets
        for met in mets:
            if met in trn_mets:
                self.metrics['Train' + met] = list()
            else:
                for handler in self.multi_handler.tst_handlers:
                    self.metrics['Test' + handler.data_name + met] = list()
        
    def make_print(self, name, ep, reses, save, data_name=None):
        if data_name is None:
            ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        else:
            ret = 'Epoch %d/%d, %s %s: ' % (ep, args.epoch, data_name, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric if data_name is None else name + data_name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '      '
        return ret
    
    def run(self):
        self.prepare_model()
        log('Model Prepared')
        stloc = 0
        if args.load_model != None:
            self.load_model()
            stloc = len(self.metrics['TrainLoss']) * args.tst_epoch - (args.tst_epoch - 1)
        for ep in range(stloc, args.epoch):
            tst_flag = (ep % args.tst_epoch == 0)
            reses = self.train_epoch()
            log(self.make_print('Train', ep, reses, tst_flag))
            if ep % 1 == 0:
                self.multi_handler.remake_initial_projections()
            if tst_flag:
                for handler in self.multi_handler.tst_handlers:
                    reses = self.test_epoch(handler.val_loader, handler)
                    # Note that this is the validation performance
                    log(self.make_print('Test', ep, reses, tst_flag, handler.data_name))
                self.save_history()
            print()
        
        for handler in self.multi_handler.tst_handlers:
            res_summary = dict()
            times = 10
            st = time.time()
            for i in range(times):
                reses = self.test_epoch(handler.tst_loader, handler)
                log(self.make_print('Test', args.epoch, reses, False, handler.data_name))
                self.add_res_to_summary(res_summary, reses)
                self.multi_handler.remake_initial_projections()
            for key in res_summary:
                res_summary[key] /= times
            log(self.make_print('AVG', args.epoch, res_summary, False, handler.data_name))
            print(time.time() - st)
        self.save_history()

    def add_res_to_summary(self, summary, res):
        for key in res:
            if key not in summary:
                summary[key] = 0
            summary[key] += res[key]

    def print_model_size(self):
        total_params = 0
        trainable_params = 0
        non_trainable_params = 0
        for param in self.model.parameters():
            tem = np.prod(param.size())
            total_params += tem
            if param.requires_grad:
                trainable_params += tem
            else:
                non_trainable_params += tem
        print(f'Total params: {total_params/1e6}')
        print(f'Trainable params: {trainable_params/1e6}')
        print(f'Non-trainable params: {non_trainable_params/1e6}')

    def prepare_model(self):
        self.model = OpenGraph()
        t.cuda.empty_cache()
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
        self.lr_scheduler = ALRS(self.opt)
        self.print_model_size()

    def train_epoch(self):
        self.model.train()
        trn_loader = self.multi_handler.trn_loader
        trn_loader.dataset.data_shuffling()
        ep_loss, ep_preloss, ep_regloss = 0, 0, 0
        steps = len(trn_loader)
        tot_samp_num = 0
        counter = [0] * len(self.multi_handler.trn_handlers)
        for i, batch_data in enumerate(trn_loader):
            if args.epoch_max_step > 0 and i >= args.epoch_max_step:
                break
            ancs, poss, negs, adj_idx = batch_data
            adj_idx = adj_idx[0]
            ancs = ancs[0].long()
            poss = poss[0].long()
            negs = negs[0].long()
            adj = self.multi_handler.trn_handlers[adj_idx].torch_adj
            if args.cache_adj == 0:
                adj = adj.to(args.devices[0])
            initial_projector = self.multi_handler.trn_handlers[adj_idx].initial_projector
            if args.cache_proj == 0:
                initial_projector = initial_projector.to(args.devices[0])
            loss, loss_dict = self.model.cal_loss((ancs, poss, negs), adj, initial_projector)
            self.opt.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10, norm_type=2)
            self.opt.step()

            sample_num = ancs.shape[0]
            tot_samp_num += sample_num
            ep_loss += loss.item() * sample_num
            ep_preloss += loss_dict['preloss'].item() * sample_num
            ep_regloss += loss_dict['regloss'].item()
            log('Step %d/%d: loss = %.3f, pre = %.3f, reg = %.3f, pos = %.3f, neg = %.3f        ' % (i, steps, loss, loss_dict['preloss'], loss_dict['regloss'], loss_dict['posloss'], loss_dict['negloss']), save=False, oneline=True)

            counter[adj_idx] += 1
            if args.proj_trn_steps > 0 and counter[adj_idx] >= args.proj_trn_steps:
                counter[adj_idx] = 0
                dice = np.random.uniform()
                if dice < 999:
                    self.multi_handler.remake_one_initial_projection(adj_idx)
                else:
                    self.multi_handler.make_one_self_initialization(self.model, adj_idx)
        ret = dict()
        ret['Loss'] = ep_loss / tot_samp_num
        ret['preLoss'] = ep_preloss / tot_samp_num
        ret['regLoss'] = ep_regloss / steps
        t.cuda.empty_cache()
        self.lr_scheduler.step(ret['Loss'])
        return ret
    
    def test_epoch(self, tst_loader, tst_handler):
        with t.no_grad():
            self.model.eval()
            ep_recall, ep_ndcg = 0, 0
            ep_tstnum = len(tst_loader.dataset)
            steps = max(ep_tstnum // args.tst_batch, 1)
            for i, batch_data in enumerate(tst_loader):
                usrs = batch_data
                numpy_usrs = usrs.numpy()
                usrs = usrs.long().to(args.devices[1])
                trn_masks = tst_loader.dataset.csrmat[numpy_usrs].tocoo()
                cand_size = trn_masks.shape[1]
                trn_masks = t.from_numpy(np.stack([trn_masks.row, trn_masks.col], axis=0)).long().cuda()
                adj = tst_handler.torch_adj
                if args.cache_adj == 0:
                    adj = adj.to(args.devices[0])
                initial_projector = tst_handler.initial_projector#.cuda()
                if args.cache_proj == 0:
                    initial_projector = initial_projector.to(args.devices[0])
                all_preds = self.model.pred_for_test((usrs, trn_masks), adj, initial_projector, cand_size, rerun_embed=False if i!=0 else True)
                _, top_locs = t.topk(all_preds, args.topk)
                top_locs = top_locs.cpu().numpy()
                recall, ndcg = self.calc_recall_ndcg(top_locs, tst_loader.dataset.tstLocs, usrs)
                ep_recall += recall
                ep_ndcg += ndcg
                log('Steps %d/%d: recall = %.2f, ndcg = %.2f          ' % (i, steps, recall, ndcg), save=False, oneline=True)
                # t.cuda.empty_cache()
        ret = dict()
        ret['Recall'] = ep_recall / ep_tstnum
        ret['NDCG'] = ep_ndcg / ep_tstnum
        t.cuda.empty_cache()
        return ret
    
    def calc_recall_ndcg(self, topLocs, tstLocs, batIds):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = 0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
            recall = dcg = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg
    
    def save_history(self):
        if args.epoch == 0:
            return
        with open('../History/' + args.save_path + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        content = {
            'model': self.model,
        }
        t.save(content, '../Models/' + args.save_path + '.mod')
        log('Model Saved: %s' % args.save_path)

    def load_model(self):
        ckp = t.load('../Models/' + args.load_model + '.mod')
        self.model = ckp['model']
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

        with open('../History/' + args.load_model + '.his', 'rb') as fs:
            self.metrics = pickle.load(fs)
        log('Model Loaded')

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if len(args.gpu.split(',')) > 1:
        args.devices = ['cuda:0', 'cuda:1']
    else:
        args.devices = ['cuda:0', 'cuda:0']
    args.devices = list(map(lambda x: t.device(x), args.devices))
    logger.saveDefault = True
    setproctitle.setproctitle('OpenGraph')

    log('Start')
    trn_datasets = ['gen1']
    tst_datasets = ['ml1m', 'ml10m', 'collab']

    # trn_datasets = ['gen2']
    # tst_datasets = ['ddi']

    # trn_datasets = ['gen0']
    # tst_datasets = ['amazon-book']

    if len(args.tstdata) != 0:
        tst_datasets = [args.tstdata]
    if len(args.trndata) != 0:
        trn_datasets = [args.trndata]
    trn_datasets = list(set(trn_datasets))
    tst_datasets = list(set(tst_datasets))
    multi_handler = MultiDataHandler(trn_datasets, tst_datasets)
    log('Load Data')

    exp = Exp(multi_handler)
    exp.run()

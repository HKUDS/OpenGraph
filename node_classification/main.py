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
from sklearn.metrics import f1_score

class Exp:
    def __init__(self, multi_handler):
        self.multi_handler = multi_handler
        self.metrics = dict()
        trn_mets = ['Loss', 'preLoss', 'Acc']
        tst_mets = ['Acc']
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

        for handler in self.multi_handler.tst_handlers:
            # reses = self.test_epoch(handler.val_loader, handler)
            # log(self.make_print('Valid', args.epoch, reses, True, handler.data_name))
            res_summary = dict()
            times = 10
            for i in range(times):
                reses = self.test_epoch(handler.tst_loader, handler)
                log(self.make_print('Test', args.epoch, reses, False, handler.data_name))
                self.add_res_to_summary(res_summary, reses)
                self.multi_handler.remake_initial_projections()
            for key in res_summary:
                res_summary[key] /= times
            log(self.make_print('AVG', args.epoch, res_summary, False, handler.data_name))
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
        self.model = OpenGraph()#.to(args.devices[1])#.cuda()
        t.cuda.empty_cache()
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
        self.lr_scheduler = ALRS(self.opt)
        self.print_model_size()
  
    def test_epoch(self, tst_loader, tst_handler):
        with t.no_grad():
            self.model.eval()
            ep_acc, ep_tot = 0, 0
            ep_tstnum = len(tst_loader.dataset)
            steps = max(ep_tstnum // args.tst_batch, 1)
            for i, batch_data in enumerate(tst_loader):
                nodes, labels = list(map(lambda x: x.long().to(args.devices[1]), batch_data))
                adj = tst_handler.torch_adj
                if args.cache_adj == 0:
                    adj = adj.to(args.devices[0])
                initial_projector = tst_handler.initial_projector
                if args.cache_proj == 0:
                    initial_projector = initial_projector.to(args.devices[0])
                preds = self.model.pred_for_node_test(nodes, adj, initial_projector, rerun_embed=False if i!=0 else True)
                if i == 0:
                    all_preds, all_labels = preds, labels
                else:
                    all_preds = t.concatenate([all_preds, preds])
                    all_labels = t.concatenate([all_labels, labels])
                hit = (labels == preds).float().sum().item()
                ep_acc += hit
                ep_tot += labels.shape[0]
                log('Steps %d/%d: hit = %d, tot = %d          ' % (i, steps, ep_acc, ep_tot), save=False, oneline=True)
                # t.cuda.empty_cache()
        ret = dict()
        ret['Acc'] = ep_acc / ep_tot
        ret['F1'] = f1_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(), average='macro')
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
        self.model = ckp['model'].to(args.devices[1])
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
    trn_datasets = tst_datasets = [args.tstdata]
    trn_datasets = list(set(trn_datasets))
    tst_datasets = list(set(tst_datasets))
    multi_handler = MultiDataHandler(trn_datasets, tst_datasets)
    log('Load Data')

    exp = Exp(multi_handler)
    exp.run()

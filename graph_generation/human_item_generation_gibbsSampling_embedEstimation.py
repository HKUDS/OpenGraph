import random
from Utils import DataGenAgent
import pickle
import os
import Exp_Utils.TimeLogger as logger
from Exp_Utils.TimeLogger import log
from Exp_Utils.Emailer import SendMail
import numpy as np
from scipy.stats import norm
import copy

class HumanItemRelationGeneration(DataGenAgent):
    def __init__(self, item_list, length_sampler, descs, hyperparams, text_embedding_dict=None):
        super(HumanItemRelationGeneration, self).__init__()

        self.item_list = item_list
        self.length_sampler = length_sampler
        self.descs = descs
        self.hyperparams = hyperparams
        self.hyperparams['item_num'] = len(self.item_list)
        self.item_to_id = dict()
        for iid, item in enumerate(item_list):
            self.item_to_id[item] = iid
        self.reject_cnt = 0
        self.item_perm = np.random.permutation(len(self.item_list))
        self.score_history = []
        self.text_embedding_dict = dict() if text_embedding_dict is None else text_embedding_dict

        # fst_text = 'Clothing, Plus size clothing, Plus size formal wear, Dresses'
        # scd_text = 'Health & Household, Household Supplies, Air fresheners, Scented beads & charms'
        # print(fst_text)
        # print(scd_text)
        # print(self.similarity(self.text_embedding(fst_text), self.text_embedding(scd_text)))
        # exit()
    
    def binvec2list(self, bin_sample_vec):
        idxs = np.reshape(np.argwhere(bin_sample_vec != 0), [-1])
        return list(map(lambda x: self.item_list[x], idxs))
    
    def list_text(self, item_list, nums):
        item_num_list = list(zip(item_list, nums))
        item_num_list.sort(key=lambda x: x[1], reverse=True)
        ret = ''
        for i, pair in enumerate(item_num_list):
            item, num = pair[0], pair[1]
            ret += '{idx}. {item}. Frequency: {num}\n'.format(idx=i, item=item, num=num)
        return ret

    def summarize(self, item_list):
        def fuse(item_list, prefix):
            for i in range(len(item_list)):
                item = item_list[i]
                if item.startswith(prefix):
                    item_list[i] = prefix
            return item_list
        def count_and_shrink(item_list):
            dic = dict()
            for item in item_list:
                if item not in dic:
                    dic[item] = 0
                dic[item] += 1
            ret_item, ret_cnt = [], []
            for key, cnt in dic.items():
                ret_item.append(key)
                ret_cnt.append(cnt)
            return ret_item, ret_cnt
        def count_prefixes_of_different_depth(item_list, max_depth):
            ret_item_list = []
            prefix_dicts = [dict() for i in range(max_depth + 1)]
            for item in item_list:
                num_idx = item.index(' #')
                tem_item = item[:num_idx]
                ret_item_list.append(tem_item)
                entities = tem_item.split(', ')
                entities = list(map(lambda entity: entity.strip(), entities))
                for depth in range(max_depth + 1):
                    if depth + 1 >= len(entities):
                        break
                    tem_prefix = ', '.join(entities[:depth + 1])
                    if tem_prefix not in prefix_dicts[depth]:
                        prefix_dicts[depth][tem_prefix] = 0
                    prefix_dicts[depth][tem_prefix] += 1
            return prefix_dicts, ret_item_list
        max_depth = len(item_list[0].split(', '))
        prefix_dicts, ret_item_list = count_prefixes_of_different_depth(item_list, max_depth)
        if len(ret_item_list) < self.hyperparams['context_limit']:
            return ret_item_list, [1] * len(ret_item_list)
        # greedy search
        flag = False
        for depth in range(max_depth, -1, -1):
            prefix_list = [(prefix, cnt) for prefix, cnt in prefix_dicts[depth].items()]
            prefix_list.sort(key=lambda x: x[1], reverse=True)
            for prefix, cnt in prefix_list:
                if cnt == 1:
                    break
                ret_item_list = fuse(ret_item_list, prefix)
                if depth != 0:
                    # adjust the counts of shallow entities
                    shrinked_prefix = ', '.join(prefix.split(', ')[:-1])
                    prefix_dicts[depth - 1][shrinked_prefix] -= cnt - 1
                if len(set(ret_item_list)) <= self.hyperparams['context_limit']:
                    flag=True
                    break
            if flag:
                return count_and_shrink(ret_item_list)
        return count_and_shrink(ret_item_list)
    
    def text_embedding(self, text):
        if text in self.text_embedding_dict:
            embeds = self.text_embedding_dict[text]
            return embeds
        print('Embedding not found!')
        print(text)
        exit()
        embedding = self.openai_embedding(text)
        self.text_embedding_dict[text] = embedding
        return embedding
    
    def similarity(self, fst_embed, scd_embed):
        # fst_embed = fst_embed / np.sqrt(np.sum(np.square(fst_embed)))
        # scd_embed = scd_embed / np.sqrt(np.sum(np.square(scd_embed)))
        return np.sum(fst_embed * scd_embed)
    
    def dynamic_normalize(self, score):
        self.score_history.append(score)
        if len(self.score_history) > 5000:
            self.score_history = self.score_history[-5000:]
        if len(self.score_history) < 5:
            return max(min(1.0, score), 0.0)
        score_samples = np.array(self.score_history)
        mean = np.mean(score_samples)
        std = np.sqrt(np.mean((score_samples - mean) ** 2))
        minn = mean - 1.96 * std
        maxx = mean + 1.96 * std#2.96 * std
        # minn = np.min(self.score_history)
        # maxx = np.max(self.score_history)
        ret = (score - minn) / (maxx - minn)
        ret = max(min(ret, 1.0), 0.0)
        # add margin
        # ret = (ret + 0.1) / 1.2
        # print('minn', minn)
        # print('maxx', maxx)
        # print('score', score)
        # print((score - minn) / (maxx - minn))

        # if len(self.score_history) > 10000:
        #     import matplotlib.pyplot as plt
        #     plt.hist(self.score_history, 100)
        #     plt.savefig('tem.pdf')
        #     exit()

        return ret

    def estimate_probability(self, bin_sample_vec, cur_dim, is_deleting=False):
        item_list = self.binvec2list(bin_sample_vec)
        new_item = self.item_list[cur_dim]
        candidate_embedding = self.text_embedding(new_item[:new_item.index(' #')])

        # summary similarity
        # if 'real_data' not in self.descs['data_name']:
        #     summarized_item_list, nums = self.summarize(item_list)
        #     summarized_score = 0
        #     for i, summarized_item in enumerate(summarized_item_list):
        #         sim = self.similarity(self.text_embedding(summarized_item), candidate_embedding)
        #         summarized_score += sim * nums[i] / sum(nums)
        
        # top instance similarity
        # sims = list(map(lambda item: (self.similarity(self.text_embedding(item[:item.index(' #')]), candidate_embedding), item[:item.index(' #')]), item_list))
        # sims.sort(reverse=True, key=lambda x: x[0])
        # sim_scores = list(map(lambda x: x[0], sims[:self.hyperparams['context_limit']]))
        # instance_score = sum(sim_scores) / len(sim_scores)
        # most_sim_items = list(map(lambda x: x[1], sims[:self.hyperparams['context_limit']]))

        embed_list = list(map(lambda item: self.text_embedding(item[:item.index(' #')]), item_list))
        avg_embed = sum(embed_list) / len(embed_list)
        instance_score = np.sum(avg_embed * candidate_embedding)


        # print('## Candidate item:', new_item)
        # print('## Summary context:')
        # print(self.list_text(summarized_item_list, nums))
        # print('## Most similar items:')
        # print(self.list_text(most_sim_items, [1] * len(most_sim_items)))
        
        # if 'real_data' in self.descs['data_name']:
        score = instance_score
        # else:
        #     score = (summarized_score + instance_score) / 2
        # log('Original score: summary-{sum_score}, instance-{ins_score}, sum-{tot_score}'.format(sum_score=summarized_score, ins_score=instance_score, tot_score=score))
        # score = self.dynamic_normalize(score)
        # log('Dynamic normalized probability: {prob}'.format(prob=score))
        interaction_num = np.sum(bin_sample_vec != 0)
        interaction_prob = 1.0 / (1.0 + np.exp((interaction_num - self.hyperparams['length_center'])/(self.hyperparams['length_center']//2)))#(100, 50), (150, 75)
        score = score * interaction_prob# * 1.1
        # log('Interaction num normalized probability: {prob}'.format(prob=score))
        return score
    
    def update_sample(self, last_sample, cur_dim, should_include):
        if last_sample[cur_dim] == 0.0 and should_include or last_sample[cur_dim] > 0.0 and not should_include:
            new_sample = copy.deepcopy(last_sample)
            new_sample[cur_dim] = 1.0 - last_sample[cur_dim]
            return new_sample, True
        else:
            return last_sample, False

    def Gibbs_Sampling(self):
        samples = []
        idx = 0
        update_cnt = 0
        cur_community = 0
        for step in range(self.hyperparams['sample_num']):
            if step % self.hyperparams['restart_step'] == 0:
                samples.append(self.random_sample())
                cur_community = (cur_community + 1) % self.hyperparams['community_num']
            last_sample = samples[-1]
            update_flag = False
            for small_step in range(self.hyperparams['gibbs_step']):
                cur_dim = self.item_perm[idx]
                nnz = np.sum(last_sample != 0)
                delete_dice = random.uniform(0, 1)
                if nnz > self.hyperparams['delete_nnz'] and delete_dice < 0.5:# 0.5, 0.4, 0.75, 2
                    cur_dim = np.random.choice(np.reshape(np.argwhere(last_sample > 0.0), [-1]))
                tem_delete_flag = False
                if last_sample[cur_dim] > 0.0:
                    # log('Deleting')
                    tem_delete_flag = True
                    last_sample[cur_dim] = 0.0
                idx = (idx + 1) % len(self.item_list)
                self.failure = 0
                prob = self.estimate_probability(last_sample, cur_dim, tem_delete_flag)
                
                # community modifier
                diff = abs(cur_community - cur_dim % self.hyperparams['community_num'])
                prob *= self.hyperparams['com_decay'] ** diff

                dice = random.uniform(0, 1) - self.hyperparams['dice_shift']
                # if dice < prob:
                #     print('Edge should be included')
                # else:
                #     print('Edge should not be included')
                last_sample, change_flag = self.update_sample(last_sample, cur_dim, dice < prob)
                if tem_delete_flag:
                    change_flag = not change_flag
                if change_flag:
                    if small_step == 0:
                        log('Sample Updated! Step {step}_{small_step}, update cnt {update_cnt}, interaction num {int_num}, sample num {samp_num}'.format(step=step, small_step=small_step, update_cnt=update_cnt, int_num=np.sum(last_sample!=0.0), samp_num=len(samples)), oneline=True)
                    self.reject_cnt = 0
                    update_cnt += 1
                    update_flag = True
                else:
                    if small_step == 0:
                        log('Sample UNCHANGED! Step {step}_{small_step}, update cnt {update_cnt}, interaction num {int_num}, sample num {samp_num}'.format(step=step, small_step=small_step, update_cnt=update_cnt, int_num=np.sum(last_sample!=0.0), samp_num=len(samples)), oneline=True)
                    self.reject_cnt += 1
                # print('*******\n')
                if self.reject_cnt > 50:
                    log('Consecutive rejection {rej_cnt} when sampling!'.format(rej_cnt=self.reject_cnt), save=True)
                    log('Last sample: {last_sample}'.format(last_sample=self.binvec2list(samples[-1])))
                    log('New sample: {new_sample}'.format(new_sample=self.binvec2list(self.update_sample(last_sample, cur_dim, dice >= prob))))
                    log('Sending report email.', save=True)
                    # SendMail(logger.logmsg)
                    self.reject_cnt = 0
                    break
            if update_flag:
                if step % self.hyperparams['restart_step'] < self.hyperparams['gibbs_skip_step']: # original 50
                    samples[-1] = last_sample
                else:
                    samples.append(last_sample)
        return samples
    
    def random_sample(self):
        picked_idxs = random.sample(list(range(len(self.item_list))), self.hyperparams['seed_num'])
        last_interaction = np.zeros(len(self.item_list))
        last_interaction[picked_idxs] = 1.0
        return last_interaction
    
    def run(self):
        samples_binvec = self.Gibbs_Sampling()
        picked_items_list = []
        for vec in samples_binvec:
            picked_items = self.binvec2list(vec)
            picked_items_list.append(picked_items)
        return picked_items_list

def load_item_list(item_file, entity_file, item_num):
    if not os.path.exists(item_file):
        with open(entity_file, 'rb') as fs:
            entity_tree_root = pickle.load(fs)
        entity_tree_root.allocate_number(item_num)
        item_list = entity_tree_root.get_list_of_leaves('')
        with open(item_file, 'wb+') as fs:
            pickle.dump(item_list, fs)
    else:
        with open(item_file, 'rb') as fs:
            item_list = pickle.load(fs)
    return item_list

def get_gen_iter(file_root, interaction_file_prefix):
    max_existing_iter = -1
    for filename in os.listdir(file_root):
        cur_filename = file_root + filename
        if interaction_file_prefix in cur_filename:
            st_idx = len(interaction_file_prefix)
            ed_idx = cur_filename.index('_iter-0.pkl')
            cur_iter = int(cur_filename[st_idx: ed_idx])
            max_existing_iter = max(max_existing_iter, cur_iter)
    return max_existing_iter

def load_interactions(prev_interaction_file):
    if 'iter--1' in prev_interaction_file:
        return None
    with open(prev_interaction_file, 'rb') as fs:
        prev_interactions = pickle.load(fs)
    return prev_interactions

def load_embedding_dict(embed_file):
    if not os.path.exists(embed_file):
        return None
    with open(embed_file, 'rb') as fs:
        ret = pickle.load(fs)
    return ret

if __name__ == '__main__':
    # parameter specification
    descs = {
        'data_name': 'gen_data_ecommerce',
        'scenario_desc': 'e-commerce platform like Amazon',
        'human_role': 'user',
        'interaction_verb': 'interact',
        'initial_entity': 'products',
    }
    hyperparams = {
        'seed_num': 6,
        'item_num': 1,#200000,
        'sample_num': 1400,#20000
        'context_limit': 15,
        'gibbs_step': 1000,
        'gen_base': 0,
        'restart_step': 100,# shift community when restart
        'gibbs_skip_step': 1,# 100
        'delete_nnz': 1, # 5
        'length_center': 400, # 60, 100, 150
        'community_num': 7,
        'itmfuse': True,
        'com_decay': 0.95,
        # 'dice_shift': 0.1,
        'dice_shift': 0.1,
    }

    # file name definition
    file_root = 'gen_results/datasets/{data_name}/'.format(data_name=descs['data_name'])
    entity_file = 'gen_results/tree_wInstanceNum_{initial_entity}_{scenario}.pkl'.format(initial_entity=descs['initial_entity'], scenario=descs['scenario_desc'])
    item_file = file_root + 'item_list.pkl'
    embed_file = file_root + 'embedding_dict.pkl'

    # load data
    item_list = load_item_list(item_file, entity_file, hyperparams['item_num'])
    if hyperparams['itmfuse']:
        item_list = list(map(lambda x: x[:x.index(' #')] + ' #1', item_list))
    embedding_dict = load_embedding_dict(embed_file)

    def length_sampler():
        min_len, max_len = hyperparams['min_len'], hyperparams['max_len']
        return random.randint(min_len, max_len)

    # generate new interactions
    generator = HumanItemRelationGeneration(item_list, length_sampler, descs, hyperparams, embedding_dict)
    sampled_interactions = generator.run()

    # store to disk
    interaction_file_prefix = file_root + 'interaction_base-'
    if 'gen_base' in hyperparams:
        gen_base = hyperparams['gen_base']
    next_interaction_file = interaction_file_prefix + str(gen_base) + '_iter-0.pkl'
    if os.path.exists(next_interaction_file):
        gen_base = get_gen_iter(file_root, interaction_file_prefix) + 1
        next_interaction_file = interaction_file_prefix + str(gen_base) + '_iter-0.pkl'
    with open(next_interaction_file, 'wb+') as fs:
        pickle.dump(sampled_interactions, fs)
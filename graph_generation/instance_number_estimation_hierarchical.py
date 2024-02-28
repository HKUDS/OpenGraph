import numpy as np
from Utils import DataGenAgent, EntityTreeConstructer, EntityTreeNode
import os
import time
import pickle
import Exp_Utils.TimeLogger as logger
from Exp_Utils.TimeLogger import log
from Exp_Utils.Emailer import SendMail

class HierarchicalInstanceNumberEstimator(DataGenAgent):
    def __init__(self, entity_tree_root, total_num, depth, initial_entity, scenario_desc):
        super(HierarchicalInstanceNumberEstimator, self).__init__()

        self.entity_tree_root = entity_tree_root
        self.total_num = total_num
        self.initial_entity = initial_entity
        self.scenario_desc = scenario_desc
        self.depth = depth
        self.failure = 0
    
    def _entity_list_to_text(self, entity_list):
        ret = ''
        for i, entity in enumerate(entity_list):
            ret += '{idx}. {entity}\n'.format(idx=i+1, entity=entity)
        return ret
    
    def interpret_one_answer(self, answer_text, subcategory):
        answer_lower = answer_text.lower()
        if subcategory.lower() not in answer_lower:
            log('ERROR: Entity name not found.', save=True)
            log('subcategory: {subcategory}'.format(subcategory=subcategory), save=True)
            log('answer_lower: {answer_lower}'.format(answer_lower=answer_lower), save=True)
            raise Exception('Entity name not found.')
        estimation_choices = ['average frequency', '1.2 times more frequent', '1.2 times less frequent', '1.5 times more frequent', '1.5 times less frequent', '2 times more frequent', '2 times less frequent', '4 times more frequent', '4 times less frequent', '8 times more frequent', '8 times less frequent']
        estimation_scores = [1.0, 1.2, 1/1.2, 1.5, 1/1.5, 2.0, 1/2.0, 4.0, 1/4.0, 8.0, 1/8.0]
        for i, choice in enumerate(estimation_choices):
            if choice in answer_lower:
                return estimation_scores[i]
        raise Exception('Estimation not found.')

    def interpret(self, answers_text, subcategories):
        answer_list = answers_text.strip().split('\n')
        assert len(answer_list) == len(subcategories), 'Length does not match.'
        answers = []
        for i in range(len(answer_list)):
            answers.append(self.interpret_one_answer(answer_list[i], subcategories[i]))
        return answers
    
    def estimate_subcategories(self, subcategories, category):
        subcategories_text = self._entity_list_to_text(subcategories)
        if category != self.initial_entity:
            text = '''In the context of {scenario_desc}, you are given a list of sub-categories below, which belong to the {category} category of {initial_entity}. Using your intuition and common sense, your goal is to identify the frequency of these sub-categories compared to the average frequency of all possible {category} {initial_entity}.'''.format(scenario_desc=self.scenario_desc, initial_entity=self.initial_entity, category=category)
        else:
            text = '''In the context of {scenario_desc}, you are given a list of sub-categories below, which belong to {initial_entity}. Using your intuition and common sense, your goal is to identify the frequency of these sub-categories compared to the average frequency of all possible {initial_entity}.'''.format(scenario_desc=self.scenario_desc, initial_entity=self.initial_entity)
        text += '''Your answer should contain one line for each of the sub-categories, EXACTLY following the following format: "[serial number]. [sub-category name same as in the input]; [your frequency estimation]; [one-sentence explaination for your estimation]". The frequency estimation should be one of the following choices: [average frequency, 1.2 times more/less frequent, 1.5 times more/less frequent, 2 times more/less frequent, 4 times more/less frequent, 8 times more/less frequent]. No other words should be included in your response. The sub-categories list is as follows:\n\n''' + subcategories_text
        # print('input')
        # print(text)
        try:
            answers_text = self.openai(text)
            print('Answers text:')
            print(answers_text)
            return self.interpret(answers_text, subcategories)
        except Exception as e:
            self.failure += 1
            if self.failure < 5:
                log('Exception occurs when interpreting. Retry in 10 seconds.', save=True)
                log('Exception message: {exception}'.format(exception=e), save=True)
                log('Failure times: {failure}'.format(failure=self.failure), save=True)
                log('Prompt text:\n{prompt}'.format(prompt=text), save=True)
                log('Response text:\n{response}'.format(response=answers_text), save=True)
                time.sleep(10)
                return self.estimate_subcategories(subcategories, category)
            else:
                log('Exception occurs {failure} times when interpreting. CANNOT HANDLE.'.format(failure=str(self.failure)), save=True)
                log('Exception message: {exception}'.format(exception=e), save=True)
                log('Prompt text:\n{prompt}'.format(prompt=text), save=True)
                log('Response text:\n{response}'.format(response=answers_text), save=True)
                log('Sending report email.', save=True)
                SendMail(logger.logmsg)
                logger.logmsg = ''
                return [1.0] * len(subcategories)
    
    def run(self):
        que = [self.entity_tree_root]
        while len(que) > 0:
            cur_entity = que[0]
            que = que[1:]
            if len(cur_entity.children) == 0:
                continue
            cur_children_entities = list(cur_entity.children.values())
            que = que + cur_children_entities
            cur_children_names = list(map(lambda x: x.entity_name, cur_children_entities))
            assert self.depth - cur_entity.depth > 0 and self.depth - cur_entity.depth < self.depth
            for _ in range(self.depth - cur_entity.depth + 1):
                self.failure = 0
                # answers = self.estimate_subcategories(cur_children_names, cur_entity.entity_name)
                # print(answers)
                # print('-----------------')
                # print()
                for j, entity in enumerate(cur_children_entities):
                    # entity.frequency.append(answers[j])
                    entity.frequency.append(1.0)
        self.entity_tree_root.allocate_number(self.total_num)
        with open('gen_results/tree_wInstanceNum_{initial_entity}_{scenario}.pkl'.format(initial_entity=self.initial_entity, scenario=self.scenario_desc), 'wb') as fs:
            pickle.dump(self.entity_tree_root, fs)


scenario = 'e-commerce platform like Amazon'
initial_entity = 'products'
total_num = 200000
depth = 5

# scenario = 'published paper list of top AI conferences'
# initial_entity = 'deep learning papers'
# total_num = 1000000
# depth = 6

# scenario = 'venue rating platform like yelp'
# initial_entity = 'business venues'
# total_num = 30000
# depth = 5

# scenario = 'book rating platform'
# initial_entity = 'books'
# total_num = 30000
# depth = 5

# load entities
file = os.path.join('gen_results/', '{entity}_{scenario}.txt'.format(entity=initial_entity, scenario=scenario))
entity_lines = []
with open(file, 'r') as fs:
    for line in fs:
        entity_lines.append(line)
entity_tree_constructer = EntityTreeConstructer(entity_lines)
entity_tree_root = entity_tree_constructer.root

estimator = HierarchicalInstanceNumberEstimator(entity_tree_root, total_num=total_num, depth=depth, initial_entity=initial_entity, scenario_desc=scenario)
estimator.run()
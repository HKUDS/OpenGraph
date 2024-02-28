import time
import openai
import json
import tiktoken
import numpy as np
import Exp_Utils.TimeLogger as logger
from Exp_Utils.TimeLogger import log
from Exp_Utils.Emailer import SendMail
import time

openai.api_key = "xx-xxxxxx"

class DataGenAgent:
    def __init__(self):
        super(DataGenAgent, self).__init__()
        self.token_num = 0
        self.encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    
    def openai_embedding(self, message):
        try:
            embedding = openai.Embedding.create(
                model='text-embedding-ada-002',
                input = message
            )['data'][0]['embedding']
            # time.sleep()
            return np.array(embedding)
        except Exception as e:
            print('OpenAI request error: {err_msg}. Retry in 10 seconds.'.format(err_msg=e))
            time.sleep(10)
            return self.openai_embedding(message)

    def openai(self, message):
        try:
            completion = openai.ChatCompletion.create(
                model='gpt-3.5-turbo-1106',
                # model='gpt-4',
                messages=[
                    {"role": "user", "content": message},
                ]
            )
            response = completion.choices[0].message.content
            time.sleep(1)
            self.token_num += len(self.encoding.encode(json.dumps(message)))
            return response
        except Exception as e:
            print('OpenAI request error: {err_msg}. Retry in 10 seconds.'.format(err_msg=e))
            time.sleep(10)
            return self.openai(message)
    
    def handling_llm_exceptions(self, message, interpret_func, interpret_args, failure_tolerance):
        try:
            answers_text = self.openai(message)
            print('Answers text:')
            print(answers_text)
            print('----------\n')
            return 0, interpret_func(answers_text, *interpret_args)
        except Exception as e:
            self.failure += 1
            log('\n**********\nERROR\n')
            log('Exception occurs when interpreting. Exception message: {exception}'.format(exception=e), save=True)
            log('Failure times: {failure}'.format(failure=self.failure, save=True))
            log('Prompt text:\n{prompt}'.format(prompt=message), save=True)
            log('Response text:\n{response}'.format(response=answers_text), save=True)
            if self.failure < failure_tolerance:
                log('Retry in 10 seconds.', save=True)
                time.sleep(10)
                log('\n**********\n')
                return 1, None
            else:
                log('Reaching maximum failure tolerance. CANNOT HANDLE!'.format(failure=self.failure), save=True)
                log('Sending report email.', save=True)
                SendMail(logger.logmsg)
                logger.logmsg = ''
                log('\n**********\n')
                return 2, None

class EntityTreeNode:
    def __init__(self, entity_name, depth, parent=None):
        self.entity_name = entity_name
        self.frequency = []
        self.quantity = -1
        self.children = dict()
        self.parent = parent
        self.depth = depth
    
    def is_child(self, entity_name):
        return entity_name in self.children
    
    def to_child(self, entity_name):
        return self.children[entity_name]

    def add_child(self, entity_name):
        child = EntityTreeNode(entity_name, self.depth+1, self)
        self.children[entity_name] = child
    
    def iterate_children(self):
        for key, node in self.children.items():
            yield key, node
    
    def allocate_number(self, quantity):
        print('Allocating depth {depth} {entity_name}, quantity: {quantity}'.format(depth=self.depth, entity_name=self.entity_name, quantity=quantity))
        self.quantity = quantity
        if len(self.children) == 0:
            return
        child_list = list(self.children.values())
        child_freq = list(map(lambda x: x.frequency, child_list))
        child_freq = np.array(child_freq) # N * T
        if child_freq.shape[1] == 0:
            raise Exception('No estimated frequency for children.')
        summ = np.sum(child_freq, axis=0, keepdims=True) # 1 * T
        child_freq = child_freq / summ # N * T
        child_num = np.mean(child_freq, axis=1) * self.quantity # N
        for i, child in enumerate(child_list):
            child.allocate_number(child_num[i])
    
    def get_list_of_leaves(self, entity_name, with_branches=False):
        if len(self.children) == 0:
            num = max(1, int(self.quantity))
            entity_list = list()
            cur_entity_name = entity_name + ', ' + self.entity_name
            for i in range(num):
                # entity_list.append(cur_entity_name + ' #{idx}'.format(idx=i))
                entity_list.append(self.entity_name + ' #{idx}'.format(idx=i))
            return entity_list
        entity_list = list()
        if with_branches:
            if self.depth <= 2:
                tem_entity_name = self.entity_name
            else:
                tem_entity_name = entity_name + ', ' + self.entity_name
            entity_list.append(tem_entity_name)
        for _, child in self.iterate_children():
            nxt_entity_name = self.entity_name if self.depth <= 2 else (entity_name + ', ' + self.entity_name)
            entities = child.get_list_of_leaves(nxt_entity_name, with_branches)
            entity_list = entity_list + entities
        return entity_list

class EntityTreeConstructer:
    def __init__(self, entity_lines):
        super(EntityTreeConstructer, self).__init__()
        
        root_name = self.line_process(entity_lines[0])[0]
        self.root = EntityTreeNode(root_name, depth=1)
        self.root.frequency.append(1.0)
        self.construct_tree(entity_lines)
    
    def add_node(self, cur_node, descriptions, cur):
        parent_entity_name = descriptions[cur-1]
        if cur_node.entity_name != parent_entity_name:
            print(cur_node.entity_name, parent_entity_name)
            print(descriptions)
        assert cur_node.entity_name == parent_entity_name
        cur_entity_name = descriptions[cur]
        if not cur_node.is_child(cur_entity_name):
            cur_node.add_child(cur_entity_name)
        if cur + 1 < len(descriptions):
            self.add_node(cur_node.to_child(cur_entity_name), descriptions, cur+1)
    
    def line_process(self, entity_line, check=False):
        entity_line = entity_line.strip()
        descriptions = list(map(lambda x: x.strip(), entity_line.split(',')))
        if not check:
            return descriptions
        if descriptions[0] != self.root.entity_name:
            raise Exception('Cannot find root')
        if len(descriptions) <= 1:
            raise Exception('Fail to split')
        return descriptions
    
    def construct_tree(self, entity_lines):
        for entity_line in entity_lines:
            try:
                descriptions = self.line_process(entity_line, check=True)
            except Exception as e:
                print(str(e), ':', entity_line)
                continue
            self.add_node(self.root, descriptions, cur=1)
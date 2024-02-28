import pickle
import os
from Utils import DataGenAgent
from Exp_Utils.TimeLogger import log

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

descs = {
    'data_name': 'gen_data_venues',
    'scenario_desc': 'venue rating platform like yelp',
    'human_role': 'user',
    'interaction_verb': 'interact',
    'initial_entity': 'business venues',
}
descs = {
    'data_name': 'gen_data_books',
    'scenario_desc': 'book rating platform',
    'human_role': 'user',
    'interaction_verb': 'interact',
    'initial_entity': 'books',
}
descs = {
    'data_name': 'gen_data_ai_papers',
    'scenario_desc': 'published paper list of top AI conferences',
    'human_role': 'user',
    'interaction_verb': 'interact',
    'initial_entity': 'deep learning papers',
}
descs = {
    'data_name': 'gen_data_ecommerce',
    'scenario_desc': 'e-commerce platform like Amazon',
    'human_role': 'user',
    'interaction_verb': 'interact',
    'initial_entity': 'products',
}
file_root = 'gen_results/datasets/{data_name}/'.format(data_name=descs['data_name'])
entity_file = 'gen_results/tree_wInstanceNum_{initial_entity}_{scenario}.pkl'.format(initial_entity=descs['initial_entity'], scenario=descs['scenario_desc'])
embed_file = file_root + 'embedding_dict.pkl'

with open(entity_file, 'rb') as fs:
    entity_tree_root = pickle.load(fs)
entity_tree_root.allocate_number(1)
item_list = entity_tree_root.get_list_of_leaves('', with_branches=True)
item_list = list(map(lambda item_name: item_name if ' #' not in item_name else item_name[:item_name.index(' #')], item_list))
print(item_list)
print('Num of items', len(item_list))
agent = DataGenAgent()
embedding_dict = dict()
for i, item in enumerate(item_list):
    log('{idx} / {tot}'.format(idx=i, tot=len(item_list)))
    embedding = agent.openai_embedding(item)
    embedding_dict[item] = embedding
with open(embed_file, 'wb') as fs:
    pickle.dump(embedding_dict, fs)
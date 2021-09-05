import json
import pickle
import numpy as np
import pandas as pd

import torch

import sys
sys.path.append('/users4/ldu/abductive/')
from learning_event_order import _truncate_seq_pair, InputFeatures, convert_examples_to_features, ini_from_pretrained
from onmt.BertModules import *
from onmt.Utils import select_field

import pdb

def sample_convert(sample):
    sample_n_ls = []
    '''
    keys = [('obs1', 'obs1'), ('obs1', 'hyp1'), ('obs1', 'obs2'), ('hyp1', 'hyp1'), ('hyp1', 'obs2'), ('obs2', 'obs2'),
            ('obs1', 'hyp2'), ('hyp2', 'hyp2'), ('hyp2', 'obs2')]
    '''
    for hyp in ['hyp1', 'hyp2']:
        keys = ['obs1',  hyp, 'obs2']
        key_ls = [(k1, k2) for k1 in keys for k2 in keys]
        '''
        for k in keys:
            sample_n = {}
            sample_n['sent1'] = sample[k[0]]
            sample_n['sent2'] = sample[k[1]]
            sample_n['ans'] = 0
            sample_n_ls.append(sample_n)
        '''
        for k in key_ls:
            sample_n = {}
            sample_n['sent1'] = sample[k[0]]
            sample_n['sent2'] = sample[k[1]]
            sample_n['ans'] = 0
            sample_n_ls.append(sample_n)

    return sample_n_ls 
    
    
def reshape(weights):
    graph = np.zeros([3, 3])
    graph_sym = np.zeros([3, 3])
    
    graph[np.triu_indices(3)] = weights
    
    graph_sym[np.triu_indices(3, 1)] = 1 - graph[np.triu_indices(3, 1)]
    graph_sym = graph_sym.T
    
    graph = graph + graph_sym 
    
    # each rowsum is scaled to 1
    graph = np.dot(np.diag(1/graph.sum(1)), graph)
    
    return graph
    
    
def graph_matching(sample, tokenizer, model):
    
    sample_converted = sample_convert(sample)
    
    features = convert_examples_to_features(sample_converted, tokenizer, max_seq_length=40,  is_training=False)
    
    input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
    segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
    
    input_ids = input_ids.squeeze().cuda(gpu_ls[0])
    segment_ids = segment_ids.squeeze().cuda(gpu_ls[0])
    
    cls_scores = model(input_ids = input_ids, token_type_ids = segment_ids)
    cls_scores = cls_scores.softmax(-1)[:, 0].detach().cpu().numpy()
    
    #weights_1 = cls_scores[:6]
    #weights_2 = cls_scores[[0, 2, 5, 6, 7, 8]]

    weights_1 = cls_scores[:9].reshape(3, 3)
    weights_2 = cls_scores[-9:].reshape(3, 3)
    
    #graph = [reshape(weights_1), reshape(weights_2)]
    graph = [weights_1, weights_2]
    
    return graph


def ini_from_pretrained(paras):

    model_config = paras[0]
    state_dict = paras[1]

    model = BertForNextSentencePrediction(model_config)
    old_keys = []
    new_keys = []
    
    for key in state_dict.keys():
        new_key = key
        if 'module.' in new_key:
            new_key = new_key.replace('module.', '')        
        if new_key != key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)
        
    for name, parameter in model.state_dict().items():
        if name in state_dict.keys():
            try:
                bert_p = state_dict[name]
                parameter.data.copy_(bert_p.data)
            except:
                print('dimension mismatch! ' + name)
        else:
            #pdb.set_trace()
            print(name)
    model.keys_bert_parameter = state_dict.keys()
    
    return model


    
tokenizer = torch.load("/users4/ldu/abductive/pretrained_models/roberta_base/tokenizer.pt")
#model_para = torch.load("/users4/ldu/abductive/pretrained_models/event_order_pretrained/0.9375e_2b_b_7_10_2_3_2_2_0.1_d_0.01_F_F_c_g_3e-05_0.1_0.04_um216358.pkl")
model_para = torch.load("/users4/ldu/abductive/pretrained_models/event_order_pretrained/20000.7916666666666666e_2gb_b_7_10_2_3_2_2_0.1_d_0.01_F_F_c_g_3e-05_0.1_0.04_um_600000_roc_sis664113.pkl")

model = ini_from_pretrained(model_para)

gpu_ls = [0, 1]

model = nn.DataParallel(model,  device_ids=gpu_ls)
model.cuda(gpu_ls[0])


datset = ['train', 'dev']

num = 0
for dat in datset:
    f = open("/users4/ldu/abductive/data/dat0/" + dat + ".jsonl")
    samples = f.readlines()
    f.close()
    
    f = open("/users4/ldu/abductive/data/dat0/" + dat + "-labels.lst")
    ans = f.readlines()
    f.close()
    
    sample_ls = []
    
    for sample, ans in zip(samples, ans):
        sample = json.loads(sample)
        
        sample_pret = {}
        sample_pret['hyps'] = [sample['hyp1'], sample['hyp2']]
        sample_pret['obs'] = [sample['obs1'], sample['obs2']]
        sample_pret['ans'] = ans
        
        sample_pret['graph'] = graph_matching(sample, tokenizer, model)
        
        sample_ls.append(sample_pret)
        
        if num % 1000 == 0:
            print(num) 
        
        num += 1
        
    f = open("/users4/ldu/abductive/data/" + dat + "_pret.pkl", 'wb')    
    pickle.dump(sample_ls, f)
    f.close()
        
 
    
    
        
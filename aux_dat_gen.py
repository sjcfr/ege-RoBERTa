import json
import pickle
import numpy as np
import pandas as pd
import random

import sys
sys.path.append('/users4/ldu/abductive/')
from learning_event_order import _truncate_seq_pair, InputFeatures, convert_examples_to_features
from onmt.BertModules import *
from onmt.Utils import select_field
from transformers import *


def sample_convert(sample):
    sample_n_ls = []
    keys = [(i, j) for i in range(len(sample['sents'])) for j in range(len(sample['sents']))]
    for k in keys:
        sample_n = {}
        sample_n['sent1'] = sample['sents'][k[0]]
        sample_n['sent2'] = sample['sents'][k[1]]
        sample_n['ans'] = 0
        sample_n_ls.append(sample_n)
    
    return sample_n_ls 
        
    
def graph_matching(sample, tokenizer, model):
    
    sample_converted = sample_convert(sample)
    
    features = convert_examples_to_features(sample_converted, tokenizer, max_seq_length=40,  is_training=False)
    
    input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
    atten_masks = torch.tensor(select_field(features, 'atten_masks'), dtype=torch.long)
    
    input_ids = input_ids.squeeze().cuda(gpu_ls[0])
    atten_masks = atten_masks.squeeze().cuda(gpu_ls[0])
    
    cls_scores = model(input_ids = input_ids, attention_mask = atten_masks)[0]
    cls_scores = cls_scores.softmax(-1)[:, 0].detach().cpu().numpy()
    
    weights = cls_scores.reshape(5, 5)
    
    graph = [weights]
    
    return graph


def ini_from_pretrained(paras):

    model_config = paras[0]
    state_dict = paras[1]

    model = RobertaForSequenceClassification(model_config)
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


random.seed(1234)

tokenizer = RobertaTokenizer.from_pretrained("/data/huggingface_transformers/roberta-large/")
model_para = torch.load("/users4/ldu/abductive/pretrained_models/event_order_pretrained/80000.9583333333333334e_2gb_l_7_10_2_3_2_2_0.1_d_0.01_F_F_c_g_3e-05_0.1_0.04_um_602000486955.pkl")
model = ini_from_pretrained(model_para)

gpu_ls = [4, 5, 6, 7]

model = nn.DataParallel(model,  device_ids=gpu_ls)
model.cuda(gpu_ls[0])

dat_ls = ['vist.csv', 'time_travel_0_keep.csv', 'activenet.csv', 'time_travel.csv']
auxdat_dict = {}

for dat in dat_ls:
    df_tmp = pd.read_csv('/users4/ldu/abductive/data/dat0/' + dat)

    sample_num = df_tmp.shape[0]
    sample_ls = []
    
    for i in range(sample_num):
        sent_ls = df_tmp.iloc[i, 1:].tolist()
        
        sample_tmp = {}
        sample_tmp['hyp1'] = sent_ls[0] 
        sample_tmp['hyp2'] = sent_ls[-1]
        sample_tmp['obs1'] = sent_ls[2]
        sample_tmp['inter1'] = sent_ls[1] 
        sample_tmp['inter2'] = sent_ls[3]
        
        sample_tmp['sents'] = sent_ls 
        sample_tmp['ans'] = 0
        
        sample_tmp['graph'] = graph_matching(sample_tmp, tokenizer, model)
        
        sample_ls.append(sample_tmp)
        
    auxdat_dict[dat] = sample_ls


sample_tot = []    
for ith, dat in enumerate(dat_ls):
    sample_tot_tmp = []
    keys_tmp = list(auxdat_dict.keys())
    keys_tmp.pop(ith)
    
    for key in keys_tmp:
        sample_tot_tmp += auxdat_dict[key]
        
    #f = open('/users4/ldu/abductive/data/aux_datset_wo' + '_' + dat + '.pkl', 'wb') 
    f = open('/users4/ldu/abductive/data/aux_datset_wo' + '_' + dat + '_1.pkl', 'wb') 
    pickle.dump(sample_tot_tmp, f)
    f.close()
    
    sample_tot += auxdat_dict[key]
            
        
#f = open('/users4/ldu/abductive/data/aux_datset.pkl', 'wb') 
f = open('/users4/ldu/abductive/data/aux_datset_1.pkl', 'wb') 
pickle.dump(sample_tot, f)
f.close()


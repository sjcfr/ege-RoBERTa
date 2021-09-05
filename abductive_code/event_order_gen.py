import random
import pickle
import copy
import json
import pandas as pd
import pdb

random.seed(1234)
sample_num = 700000
aux_datset = 'tmt_sis'


dataset = ['vist.csv', 'time_travel_0_keep.csv', 'activenet.csv', 'time_travel.csv']

df = pd.DataFrame()

for dat in dataset:
    df_tmp = pd.read_csv('/users4/ldu/abductive/data/dat0/' + dat)
    df = df.append(df_tmp)

df = df.iloc[:, 1:]
#pdb.set_trace()
sample_ls = []

for i in range(sample_num - 100000):
    label = random.randint(0, 1)
    ith_sample = random.randint(0, df.shape[0] - 1)
    
    sentence_id_ls = [n for n in range(0, 5)]
    sentence_id = sorted(random.sample(sentence_id_ls, 2))
    
    sample_tmp = {}
    
    if label == 1:
        # (sent1: pre; sent2: post)
        sample_tmp['sent1'] = df.iloc[ith_sample, sentence_id[0]]
        sample_tmp['sent2'] = df.iloc[ith_sample, sentence_id[1]]
    else:
        # (sent1: post; sent2: pre)
        sample_tmp['sent1'] = df.iloc[ith_sample, sentence_id[1]]
        sample_tmp['sent2'] = df.iloc[ith_sample, sentence_id[0]]    
            
    sample_tmp['ans'] = label
    sample_ls.append(sample_tmp)
    
    
f = open("/users4/ldu/abductive/data/dat0/anli/train.jsonl")
abductive_dat = f.readlines()
f.close()
    
abductive_dat = [json.loads(x)  for x in abductive_dat]

f = open("/users4/ldu/abductive/data/dat0/anli/train-labels.lst")
label_abductive = f.readlines()
f.close()

abductive_sample_ls = []

for i in range(102000):
    label = random.randint(0, 1)
    ith_sample = random.randint(0, len(abductive_dat) - 1)
    
    sample_tmp = abductive_dat[ith_sample]
    ans = int(label_abductive[ith_sample]) - 1
    
    event_chain = [sample_tmp['obs1'], [sample_tmp['hyp1'], sample_tmp['hyp2']][ans], sample_tmp['obs2']]
    
    abductive_sample_tmp = {}
    sent_id = random.sample([[0, 1], [1, 2]], 1)[0]
    if label == 1:
        abductive_sample_tmp['sent1'] = event_chain[sent_id[0]]
        abductive_sample_tmp['sent2'] = event_chain[sent_id[1]]
    else:
        abductive_sample_tmp['sent1'] = event_chain[sent_id[1]]
        abductive_sample_tmp['sent2'] = event_chain[sent_id[0]]
    abductive_sample_tmp['ans'] = label
    abductive_sample_ls.append(abductive_sample_tmp)
    
sample_ls = sample_ls + abductive_sample_ls    
pdb.set_trace()
#sample_ls.extend(abductive_sample_ls[:-2000])
dev_sample_ls = abductive_sample_ls[-2000:]

sample_num = len(sample_ls)


f = open('/users4/ldu/abductive/data/sentences_order_train_' + str(sample_num) + '.pkl', 'wb') 
pickle.dump(sample_ls, f)
f.close()  

#f = open('/users4/ldu/abductive/data/sentences_order_dev_' + str(sample_num) + '_' + aux_datset + '.pkl', 'wb') 
#pickle.dump(dev_sample_ls, f)
#f.close()   
    
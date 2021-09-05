
import pandas as pd
import logging
import os, sys
import argparse
import random
from tqdm import tqdm, trange
import xml.etree.ElementTree as ET
from pprint import pprint
import random
import time
import numpy as np
import pickle
import torch 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import RobertaTokenizer
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from onmt.BertModules import *
from onmt.GraphBert import *
from onmt.VariationalGraphBert import *
from onmt.Utils import *
import onmt.Opt

import pdb
sys.path.append("/users4/ldu/git_clones/apex/")
from apex import amp

'''
def loss_graph(appro_matrix, true_graph, loss_fn):
    appro_matrix = appro_matrix.squeeze()
    true_graph = true_graph.squeeze()
    assert appro_matrix.shape == true_graph.shape
    L = appro_matrix.shape[1]
    loss_tot = 0
    for i in range(L):        
        for j in range(L):
            if i != j:
                p = appro_matrix[:,i, j].unsqueeze(1)
                #p_comple = 1 - p
                p_comple = appro_matrix[:,j, i].unsqueeze(1)
                p = torch.cat([p, p_comple], axis=1)
                #p = torch.log(p) / (1 - torch.log(p))
                
                q = true_graph[:,i, j]
                
                loss_tmp = loss_fn(p, q)
                loss_tot = loss_tot + loss_tmp
        
    return loss_tot
'''
#os.environ['CUDA_VISIBLE_DEVICES']="6,7"
        
    
def mask_tokens(inputs, tokenizer):
    inputs_0 = inputs.clone()
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, 0.15)
    
    #pdb.set_trace()
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long).cuda(inputs.device)
    #random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    #pdb.set_trace()
    return inputs, labels
    
    
def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 baseline=False, voc=None):
    features_p = []
    features_q = []
    
    if 'graph' in examples[0].keys():
        has_graph = True
    else:
        has_graph = False
    num_not_append = 0
    if opt.pretrain_method == 'V':
        #examples = examples[50201:] # deleting vist examples
        examples = examples[:-50000] # deleting vist examples
    if opt.pretrain_method == 'T':
        #examples = examples[:-39590] # deleting timetravel examples
        examples = examples[-50000:] # deleting timetravel examples
        
    if opt.pretrain_method == 'R':
        examples = random.sample(examples, opt.pretrain_number)
        
    for example_index, example in enumerate(examples):
        
        p_sents = [example['hyp1'], example['obs1'], example['hyp2']]
        q_sents = example['sents']
        
        if opt.pretrain_method == 'I':
            
            random_example_1 = random.sample(examples, 1)
            random_example_2 = random.sample(examples, 1)
            
            random_sent_1 = random.sample(random_example_1[0]['sents'], 1)[0]
            random_sent_2 = random.sample(random_example_2[0]['sents'], 1)[0]
            
            q_sents[1] = random_sent_1
            q_sents[3] = random_sent_2

        if_append = True
        
        for sents, features in zip([p_sents, q_sents], [features_p, features_q]):
            
            if_append = True
            choices_features = []
            
            chain_tokens_tmp = []
            sentence_ind_tmp = []
            
            l_sents = [l for l in range(len(sents))]
            l_sents.append(-1)
        
            for ith_sent, sent in enumerate(sents):
                
                sent_tokens = tokenizer.tokenize(sent)
                chain_tokens_tmp.append(sent_tokens)
                sentence_ind_tmp.extend([ith_sent] * (len(sent_tokens) + 1))                

            tokens_tmp = [["[CLS]"]] + [token + ["[SEP]"] for token in chain_tokens_tmp]
            tokens_tmp[-1].pop()
            
            tokens_tmp = [token for tokens in tokens_tmp for token in tokens]                                           
            input_ids_tmp = tokenizer.convert_tokens_to_ids(tokens_tmp)
            input_mask_tmp = [1] * len(input_ids_tmp)

            if (max_seq_length - len(input_ids_tmp)) >= 0:
                padding = [0] * (max_seq_length - len(input_ids_tmp))
                input_ids_tmp += padding
                input_mask_tmp += padding
                sentence_ind_tmp += [p-1 for p in padding]
            else:
                input_ids_tmp = input_ids_tmp[:max_seq_length]
                input_mask_tmp = input_mask_tmp[:max_seq_length]
                sentence_ind_tmp = sentence_ind_tmp[:max_seq_length]
            
            if has_graph:
                graph = example['graph']
            else:
                graph = None
                
            if opt.pretrain_method == 'A':
                graph = [np.random.rand(5, 5)]
            
            try:
                assert len(input_ids_tmp) == max_seq_length
                assert len(input_mask_tmp) == max_seq_length
                assert len(sentence_ind_tmp) == max_seq_length
            except:
                pdb.set_trace()
            
            if set(sentence_ind_tmp) != set(l_sents[:-1]) and set(sentence_ind_tmp) != set(l_sents): 
                if_append = False
                num_not_append += 1 
                print(num_not_append)
                print("Too long example, id:", example_index)
            
            choices_features.append((tokens_tmp, input_ids_tmp, input_mask_tmp, sentence_ind_tmp, graph))
            
            answer = [0]
            try:
                answer[int(example['ans'])-1] = 1
            except:
                pdb.set_trace()
                answer[example['answer']] = 1
            
            features.append(
                InputFeatures(
                    example_id = example_index,
                    choices_features = choices_features,
                    answer = answer
                )
            )
        if not if_append:
            features_p.pop()
            features_q.pop()
    assert len(features_p) == len(features_q)
    #pdb.set_trace()
    return [features_p, features_q]    
                
                
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6,7'

parser = argparse.ArgumentParser(
    description='Train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# onmt.opts.py

onmt.Opt.model_opts(parser)
opt = parser.parse_args()


gpu_ls = parse_gpuid(opt.gpuls)

if 'large' in opt.bert_model:
    opt.train_batch_size = 12 * len(gpu_ls)
else:
    opt.train_batch_size = 18 * len(gpu_ls)

random.seed(opt.seed)
np.random.seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)

wkdir = "/users4/ldu/abductive"
os.makedirs(opt.output_dir, exist_ok=True)


train_examples = None
eval_examples = None
eval_size= None
num_train_steps = None

train_examples = load_examples(os.path.join(opt.train_data_dir))[:1000]
pdb.set_trace()
num_train_steps = int(len(train_examples) / opt.train_batch_size / opt.gradient_accumulation_steps * opt.num_train_epochs)
    
# Prepare tokenizer
#tokenizer = torch.load(opt.bert_tokenizer)
tokenizer = RobertaTokenizer.from_pretrained(opt.bert_tokenizer)

# Prepare model

model = ini_from_pretrained(opt)
  
# Prepare optimizer
if opt.fp16:
    param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                        for n, param in model.named_parameters()]
elif opt.optimize_on_cpu:
    param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                        for n, param in model.named_parameters()]
else:
    param_optimizer = list(model.named_parameters())
    
#no_decay = ['bias', 'gamma', 'beta']
#no_decay = ['gamma', 'beta']
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': opt.l2_reg},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
t_total = num_train_steps
if opt.local_rank != -1:
    t_total = t_total // torch.distributed.get_world_size()
    
optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=opt.learning_rate,
                     warmup=opt.warmup_proportion,
                     t_total=t_total)
# optimizer = adabound.AdaBound(optimizer_grouped_parameters, lr=opt.learning_rate, final_lr=0.1)

model.cuda(gpu_ls[0])

if 'large' in opt.bert_model:
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
else:
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
model_config = model.config
model = nn.DataParallel(model,  device_ids=gpu_ls)
model.config = model_config


global_step = 0

if opt.pret:
    train_features_p, train_features_q = convert_examples_to_features(
        train_examples, tokenizer, opt.max_seq_length, True)
else:
    train_features = train_examples

logger.info("***** Running training *****")
logger.info("  Num examples = %d", len(train_examples))
logger.info("  Batch size = %d", opt.train_batch_size)
logger.info("  Num steps = %d", num_train_steps)

all_example_ids = torch.tensor([train_feature_p.example_id for train_feature_p in train_features_p], dtype=torch.long)
all_input_ids_p = torch.tensor(select_field(train_features_p, 'input_ids'), dtype=torch.long)
all_attn_msks_p = torch.tensor(select_field(train_features_p, 'input_mask'), dtype=torch.long)
all_sentence_inds_p = torch.tensor(select_field(train_features_p, 'sentence_ind'), dtype=torch.long)

all_input_ids_q = torch.tensor(select_field(train_features_q, 'input_ids'), dtype=torch.long)
all_attn_msks_q = torch.tensor(select_field(train_features_q, 'input_mask'), dtype=torch.long)
all_sentence_inds_q = torch.tensor(select_field(train_features_q, 'sentence_ind'), dtype=torch.long)


all_graphs = select_field(train_features_p, 'graph') ##
all_graphs = torch.tensor(all_graphs, dtype=torch.float) ##

train_data = TensorDataset(all_example_ids, all_input_ids_p, all_attn_msks_p, all_sentence_inds_p, all_input_ids_q, all_attn_msks_q, all_sentence_inds_q, all_graphs)
if opt.local_rank == -1:
    train_sampler = RandomSampler(train_data)
else:
    train_sampler = DistributedSampler(train_data)

train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=opt.train_batch_size)

loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)

loss_aa_fn = torch.nn.CrossEntropyLoss()
Lambda = opt.Lambda
Lambda_kl = opt.Lambda_kl
loss_aa_smooth_term = opt.loss_aa_smooth
 
name = parse_opt_to_name(opt)
print(name)
time_start = str(int(time.time()))[-6:]

#test_examples_all = load_examples(os.path.join(opt.test_data_dir))
#test_features_all = convert_examples_to_features(test_examples_all, tokenizer, opt.max_seq_length, True)

#for epoch in range(int(opt.num_train_epochs / 10)):
for epoch in range(opt.num_train_epochs):
    print("Epoch:",epoch)
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
   
    for step, batch in enumerate(train_dataloader):
        pdb.set_trace()
        model.train()
        batch = tuple(t.cuda(gpu_ls[0]) for t in batch)
        '''
        for both multiple choice problem and next sentence prediction, 
        the input is context and one of the choice. 
        '''
        example_ids, input_ids_p, attn_msks_p, sentence_inds_p, input_ids_q, attn_msks_q, sentence_inds_q, graphs = batch
        num_choices = input_ids_p.shape[1]
        
        accurancy = None
        
        for n in range(num_choices):
            input_ids_p_tmp = input_ids_p[:,n,:]
            input_ids_q_tmp = input_ids_q[:,n,:]

            attn_msks_p_tmp = attn_msks_p[:,n,:]
            attn_msks_q_tmp = attn_msks_q[:,n,:]
            
            sentence_inds_p_tmp = sentence_inds_p[:,n,:]
            sentence_inds_q_tmp = sentence_inds_q[:,n,:]
            
            graphs_tmp = graphs[:,n,:]
            
            #graphs_tmp_scaled = graphs_tmp
            graphs_tmp_scaled = graphs_tmp / graphs_tmp.sum(3).unsqueeze(3)
            
            input_ids_p_tmp_msk, labels_p = mask_tokens(input_ids_p_tmp, tokenizer)
            input_ids_q_tmp_msk, _ = mask_tokens(input_ids_q_tmp, tokenizer)
            
            if opt.model_type == 'vgb':                
                pred_tokens, z_p, z_q, attn_scores = model(input_ids_p = input_ids_p_tmp_msk, input_ids_q = input_ids_q_tmp_msk, 
                                                      sentence_inds_p = sentence_inds_p_tmp, sentence_inds_q = sentence_inds_q_tmp) ##
            elif opt.model_type == 'vgb_c': 
                pred_tokens, z_p, z_q, attn_scores = model(input_ids_p = input_ids_p_tmp_msk, input_ids_q = input_ids_q_tmp_msk,graph=graphs_tmp_scaled,
                                                      sentence_inds_p = sentence_inds_p_tmp, sentence_inds_q = sentence_inds_q_tmp) ##
                                                  
            cov = torch.ones_like(z_p).cuda(input_ids_p_tmp.device)
            p = torch.distributions.normal.Normal(z_p, cov)
            q = torch.distributions.normal.Normal(z_q, cov)
                                                  
            masked_lm_loss = loss_fct(pred_tokens.view(-1, model.config.vocab_size), input_ids_p_tmp_msk.view(-1))
                
            #graphs_tmp_n = np.zeros(graphs_tmp.shape) + np.triu(np.ones(graphs_tmp.shape), 1)
            
            if opt.model_type == 'vgb':
                graphs_tmp_n = np.zeros(graphs_tmp.shape)
                
                for i in range(graphs_tmp_n.shape[2] - 1):
                    #graphs_tmp_n[:, i, i] = 1
                    graphs_tmp_n[:, :, i, i+1] = 1
                graphs_tmp_n = torch.LongTensor(graphs_tmp_n)
                #graphs_tmp = graphs_tmp_n.cuda(gpu_ls[0])
                
                '''
                for i in range(graphs_tmp_n.shape[2]):
                    for j in range(graphs_tmp_n.shape[2]):
                        if i == j:
                            graphs_tmp_n[:, :, i, j] = 0.5
                        if (j - i) == 1:
                            graphs_tmp_n[:, :, i, j] = 1
                        if (j - i) == 2:
                            graphs_tmp_n[:, :, i, j] = 0.3
                        if (j - i) == 3:
                            graphs_tmp_n[:, :, i, j] = 0.1
                #pdb.set_trace()
                graphs_tmp_n = torch.FloatTensor(graphs_tmp_n)            
                graphs_tmp = graphs_tmp_n.cuda(gpu_ls[0])
                #pdb.set_trace()    
                '''
                try:
                    loss_aa = loss_graph(attn_scores, graphs_tmp, loss_aa_fn)
                
                    loss_kl = torch.distributions.kl.kl_divergence(p, q).sum()
                    #loss = masked_lm_loss
                    loss = masked_lm_loss + Lambda * loss_aa + Lambda_kl * loss_kl
                    #loss = Lambda * loss_aa + Lambda_kl * loss_kl
                except:
                    loss = masked_lm_loss + Lambda_kl * loss_kl
                    
                if step % 20 == 0:
                    print("step:", step, "loss_msk_lm:", masked_lm_loss.detach().cpu().numpy(), "loss_aa:",loss_aa.detach().cpu().numpy() * Lambda, 'loss_kl:', Lambda_kl * loss_kl)
                
                f = open(wkdir + '/records/graph_pretrained/' + name + '_' + time_start + '.csv', 'a+')
                f.write(str(masked_lm_loss.detach().cpu().numpy()) + ',' + str(Lambda * loss_aa.detach().cpu().numpy()) + ',' + str(Lambda_kl * loss_kl.detach().cpu().numpy()) + '\n')
                f.close()

            elif opt.model_type == 'vgb_c':
                loss_kl = torch.distributions.kl.kl_divergence(p, q).sum()
                loss = masked_lm_loss + Lambda_kl * loss_kl
                
                if step % 20 == 0:
                    print("step:", step, "loss_msk_lm:", masked_lm_loss.detach().cpu().numpy(), 'loss_kl:', Lambda_kl * loss_kl)
                
                f = open(wkdir + '/records/graph_pretrained/' + name + '_' + time_start + '.csv', 'a+')
                f.write(str(masked_lm_loss.detach().cpu().numpy()) + ',' + str(Lambda_kl * loss_kl.detach().cpu().numpy()) + '\n')
                f.close()
            
            #if step > 300:
            #pdb.set_trace()
            #x=zip([p.grad.norm().detach().cpu().numpy().tolist() for p in model.parameters()],model.state_dict().keys())

            if opt.fp16 and opt.loss_scale != 1.0:
                loss = loss * opt.loss_scale
            if opt.gradient_accumulation_steps > 1:
                loss = loss / opt.gradient_accumulation_steps
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            
            #loss.backward()
            #tr_loss += loss.item()
            #nb_tr_examples += input_ids.size(0)
            #nb_tr_steps += 1
            if (step + 1) % opt.gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
                global_step += 1
                
    ls = [model.config, model.state_dict()]       
    if opt.pretrain_method in ['R', 'V', 'T']:   
        torch.save(ls, wkdir + "/pretrained_models/graph_pretrained/datset/" + "e_"  + str(epoch) + str(step) + name + time_start + '.pkl')
    elif opt.pretrain_method == 'L':   
        torch.save(ls, wkdir + "/pretrained_models/graph_pretrained/lambda/" + "e_"  + str(epoch) + str(step) + name + time_start + '.pkl')
    else:
        torch.save(ls, wkdir + "/pretrained_models/graph_pretrained/" + "e_"  + str(epoch) + str(step) + name + time_start + '.pkl')
    '''
    if epoch == 1:
        torch.save(ls, wkdir + "/ablation_models/" + "e_"  + str(epoch) + str(step) + name + time_start + '.pkl')
    if epoch > 1:
        break
    '''
        

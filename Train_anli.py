
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

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from onmt.BertModules import *
from onmt.GraphBert import *
from onmt.Utils import *
import onmt.Opt
from transformers import RobertaTokenizer

sys.path.append("/users4/ldu/git_clones/apex/")
from apex import amp

os.environ['CUDA_VISIBLE_DEVICES']="1,2,3,4,5,6,7"

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
    opt.train_batch_size = 24 * len(gpu_ls)
    #opt.train_batch_size = 8 * len(gpu_ls)

random.seed(opt.seed)
np.random.seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)

wkdir = "/users4/ldu/abductive"
os.makedirs(opt.output_dir, exist_ok=True)


train_examples = None
eval_examples = None
eval_size= None
num_train_steps = None

train_examples = load_examples(os.path.join(opt.train_data_dir))

# num_train_steps = int(len(train_examples) / opt.train_batch_size / opt.gradient_accumulation_steps * opt.num_train_epochs)
num_train_steps = int(len(train_examples) / opt.train_batch_size / opt.gradient_accumulation_steps * opt.num_train_epochs) * 2
      
# Prepare tokenizer
#tokenizer = torch.load(opt.bert_tokenizer)
tokenizer = RobertaTokenizer.from_pretrained(opt.bert_tokenizer)

# Prepare model

model = ini_from_pretrained(opt)

model_config = model.config
#model = nn.DataParallel(model,  device_ids=gpu_ls)
#model.config = model_config
model.cuda(gpu_ls[0])

if model.config.num_hidden_layers == 12:
    model_size = 'small'
elif model.config.num_hidden_layers == 24:
    model_size = 'large'

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
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

model_config = model.config
model = nn.DataParallel(model,  device_ids=gpu_ls)
model.config = model_config

global_step = 0

if opt.pret:
    train_features = convert_examples_to_features(
        train_examples, tokenizer, opt.max_seq_length, True)
else:
    train_features = train_examples

logger.info("***** Running training *****")
logger.info("  Num examples = %d", len(train_examples))
logger.info("  Batch size = %d", opt.train_batch_size)
logger.info("  Num steps = %d", num_train_steps)

all_example_ids = torch.tensor([train_feature.example_id for train_feature in train_features], dtype=torch.long)
all_input_tokens = select_field(train_features, 'tokens')
all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
all_input_masks = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
#all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
all_sentence_inds = torch.tensor(select_field(train_features, 'sentence_ind'), dtype=torch.long)

all_graphs = select_field(train_features, 'graph') ##
if all_graphs[0][0] is not None:
    all_graphs = torch.tensor(all_graphs, dtype=torch.float) ##

all_answers = torch.tensor([f.answer for f in train_features], dtype=torch.long)
train_data = TensorDataset(all_example_ids, all_input_ids, all_input_masks, all_sentence_inds, all_graphs, all_answers)
if opt.local_rank == -1:
    train_sampler = RandomSampler(train_data)
else:
    train_sampler = DistributedSampler(train_data)
# train_sampler = SequentialSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=opt.train_batch_size)

if not opt.do_margin_loss:
    loss_nsp_fn = torch.nn.CrossEntropyLoss()
else:
    loss_nsp_fn = torch.nn.MarginRankingLoss(opt.margin)
loss_aa_fn = torch.distributions.kl.kl_divergence
Lambda = opt.Lambda
loss_aa_smooth_term = opt.loss_aa_smooth
 
best_eval_acc=0.0
best_test_acc=0.0
best_step=0
eval_acc_list=[]
# for _ in trange(int(opt.num_train_epochs), desc="Epoch"):

opt.start_layer = model.config.start_layer
opt.merge_layer = model.config.merge_layer
opt.pretrain_method = model.config.pretrain_method
opt.pretrain_number = model.config.pretrain_number
            
name = parse_opt_to_name(opt)
#name = '_rand_' + name
#name = opt.bert_model.replace('/pretrained_models/graph_pretrained/','').replace('.pkl', '')
print(name)
time_start = str(int(time.time()))[-6:]

test_examples_all = load_examples(os.path.join(opt.test_data_dir))
test_features_all = convert_examples_to_features(test_examples_all, tokenizer, opt.max_seq_length, True)

f = open(wkdir + '/records/' + name + '_' + time_start + '.csv', 'a+')
f.write(opt.bert_model + '\n')
f.close()

loss_ls = []
cls_ls = []
ans_ls = []
accu_step = 0
# the effect of linear weight decay
#for epoch in range(int(opt.num_train_epochs / 10)): 
norm_ls = []
accu_0 = 0.5
accurancy = None
for epoch in range(opt.num_train_epochs):
    print("Epoch:",epoch)
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
    #freeze_params(model, requires_grad=True)
    #if epoch < opt.num_frozen_epochs:
    #    freeze_params(model, requires_grad=False)
    if epoch > 0:                

        test_features = test_features_all

        opt.test_batch_size = opt.train_batch_size
        #all_example_ids = [train_feature.example_id for train_feature in train_features]
        all_input_ids = torch.tensor(select_field(test_features, 'input_ids'), dtype=torch.long)
        all_input_masks = torch.tensor(select_field(test_features, 'input_mask'), dtype=torch.long)
        all_sentence_inds = torch.tensor(select_field(test_features, 'sentence_ind'), dtype=torch.long)
        all_answers = torch.tensor([f.answer for f in test_features], dtype=torch.long)

        all_graphs = select_field(test_features, 'graph') ##

        if all_graphs[0][0] is not None:
            all_graphs = torch.tensor(all_graphs, dtype=torch.float) ##
            test_data = TensorDataset(all_input_ids, all_input_masks, all_sentence_inds, all_graphs, all_answers)
        else:
            test_data = TensorDataset(all_input_ids, all_input_masks, all_sentence_inds, all_answers)

        # Run prediction for full data
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=opt.eval_batch_size)

        model = model.eval()
        accurancy, preds = do_evaluation(model, test_dataloader, opt, gpu_ls)

        print('step:', step, "accurancy:", accurancy)

        model = model.train()
        
        ls = [model.config, model.state_dict()]
        torch.save(ls, wkdir + "/models/" + str(step) + str(accurancy) + "e_" + str(epoch) + name + time_start + '.pkl')
        if model_size == 'large':  
            torch.save(preds, wkdir + "/results/" + str(step) + str(accurancy) + "e_" + str(epoch) + name + time_start + '.pkl')

                    
    for step, batch in enumerate(train_dataloader):
        accu_step += 1
        model.train()
        batch = tuple(t.cuda(gpu_ls[0]) for t in batch)
        '''
        for both multiple choice problem and next sentence prediction, 
        the input is context and one of the choice. 
        '''
        example_ids, input_ids, input_masks, sentence_inds, graphs, answers = batch
        num_choices = input_ids.shape[1]
        
        for n in range(num_choices):
            input_ids_tmp = input_ids[:,n,:]
            input_masks_tmp = input_masks[:,n,:]
            sentence_inds_tmp = sentence_inds[:,n,:]
            graphs_tmp = graphs[:,n,:]
            answers_tmp = answers[:,n]
            
            graphs_tmp_scaled = graphs_tmp
            if opt.model_type not in ['pb',  'bt']:
                if opt.model_type == 'gb':
                      pred_tokens, cls_scores, attn_scores = model(input_ids = input_ids_tmp, 
                                                            sentence_inds = sentence_inds_tmp, 
                                                            graphs = graphs_tmp_scaled) ##
                elif opt.model_type in ['vgb', 'vgb_c']:
                    cls_scores, attn_scores = model(input_ids_p = input_ids_tmp, attn_mask_p=input_masks_tmp,
                                                          sentence_inds_p = sentence_inds_tmp) ##   
                if not opt.do_margin_loss:
                    loss_nsp = loss_nsp_fn(cls_scores, answers_tmp)
                else:
                    #pdb.set_trace()
                    cls_scores = cls_scores.softmax(-1)
                    answers_tmp = answers_tmp.type(torch.FloatTensor).cuda(gpu_ls[0])
                    answers_tmp = (answers_tmp * 2 - 1)
                    
                    loss_nsp = loss_nsp_fn(cls_scores[:,0], cls_scores[:,1], answers_tmp)
                
                graphs_tmp_n = np.ones(graphs_tmp.shape) + np.triu(np.ones(graphs_tmp.shape) * 100, 1)
                
                loss_aa = loss_graph(attn_scores, graphs_tmp, loss_aa_fn, smooth_term=loss_aa_smooth_term)

                if opt.model_type in ['vgb', 'vgb_c']:
                    loss = loss_nsp
                else:
                    loss = loss_nsp + Lambda * loss_aa
                
                
                if step % 20 == 0:
                    print("step:", step, "loss_nsp:", loss_nsp.detach().cpu().numpy(), "loss_aa:",loss_aa.detach().cpu().numpy() * Lambda)

                f = open(wkdir + '/records/' + name + '_' + time_start + '.csv', 'a+')
                f.write(str(loss_nsp.detach().cpu().numpy()) + ',' + str(Lambda * loss_aa.detach().cpu().numpy()) + ',' + str(accurancy) + '\n')
                f.close()
                
            else:
                #pdb.set_trace()
                cls_scores = model(input_ids = input_ids_tmp) ##
                if not opt.do_margin_loss:
                    loss = loss_nsp_fn(cls_scores, answers_tmp)
                else:
                    
                    cls_scores = cls_scores.softmax(-1)
                    answers_tmp = answers_tmp.type(torch.FloatTensor).cuda(gpu_ls[0])
                    answers_tmp = (answers_tmp * 2 - 1)
                    loss = loss_nsp_fn(cls_scores[:,0], cls_scores[:,1], answers_tmp)
                                    
                f = open(wkdir + '/records/' + name + '_' + time_start + '.csv', 'a+')
                f.write(str(loss.detach().cpu().numpy()) + ',' + str(accurancy) + '\n')
                f.close()
                
                if step % 20 == 0:
                    print("step:", step, "loss:", loss.detach().cpu().numpy())

            if opt.fp16 and opt.loss_scale != 1.0:
                loss = loss * opt.loss_scale
            if opt.gradient_accumulation_steps > 1:
                loss = loss / opt.gradient_accumulation_steps
            #loss.backward()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            
            norm_ls.append([p.grad.norm().detach().cpu().numpy().tolist() for p in model.parameters()])
            
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % opt.gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
                global_step += 1
      
    ls = [model.config, model.state_dict()]
    torch.save(ls, wkdir + "/models/" + str(step) + str(accurancy) + "e_" + str(epoch) + name + time_start + '.pkl')

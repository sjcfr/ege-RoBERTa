
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
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import *
#from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
#from onmt.BertModules import *
#from onmt.GraphBert import *
from onmt.Utils import *
import onmt.Opt

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            L = len(tokens_a)
            #r = random.randint(0, L - 1)
            tokens_a.pop()
            #tokens_a.pop(0)
        else:
            L = len(tokens_b)
            #r = random.randint(0, L - 1)
            tokens_b.pop()


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 answer

    ):
        self.example_id = example_id
        self.choices_features = [
            {
                'tokens': tokens,
                'input_ids': input_ids,
                'segment_ids': segment_ids,
                'atten_masks': atten_masks
            }
            for tokens, input_ids, segment_ids, atten_masks in choices_features
        ]   
        self.answer = answer


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training, 
                                 baseline=False, voc=None):
    features = []

    num_not_append = 0
    for example_index, example in enumerate(examples):
        try:
            sent_tokens1 = tokenizer.tokenize(example['sent1'])
            sent_tokens2 = tokenizer.tokenize(example['sent2'])
        except:
            pdb.set_trace()

        choices_features = []
        
        if_append = True
               
        _truncate_seq_pair(sent_tokens1, sent_tokens2, max_seq_length - 3)
        
        tokens = ["<s>"] + sent_tokens1 + ["</s>"] + sent_tokens2 + ["</s>"]

        segment_ids = [0] * (len(sent_tokens1) + 1) + [1] * (len(sent_tokens2) + 2)
                
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        atten_masks = [1] * len(input_ids)
        
        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        segment_ids += padding
        input_ids += padding
        atten_masks += padding
        
        try:
            assert len(input_ids) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(atten_masks) == max_seq_length
        except:
            pdb.set_trace()
        
        if len(sent_tokens2) == 0:               
            if_append = False
            num_not_append += 1 
            print(num_not_append)
            print("Too long example, id:", example_index)
        
        choices_features.append((tokens, input_ids, segment_ids, atten_masks))

        answer = example['ans']
        
        if if_append:
            features.append(
                InputFeatures(
                    example_id = example_index,
                    choices_features = choices_features,
                    answer = answer
                )
            )

    return features
    
    
def ini_from_pretrained(config):
            
    bert_model = RobertaForSequenceClassification.from_pretrained(config.bert_model)
    
    return bert_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
      description='Train.py',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # onmt.opts.py
    
    onmt.Opt.model_opts(parser)
    opt = parser.parse_args()
    
    gpu_ls = parse_gpuid(opt.gpuls)
    
    if 'large' in opt.bert_model:
      opt.train_batch_size = 18 * len(gpu_ls)
    else:
      opt.train_batch_size = 32 * len(gpu_ls)
    
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    
    wkdir = "/users4/ldu/abductive/records/learning_event_order/"
    os.makedirs(opt.output_dir, exist_ok=True)
    
    
    train_examples = None
    eval_examples = None
    eval_size= None
    num_train_steps = None
    
    train_examples = load_examples(os.path.join(opt.train_data_dir))
    
    num_train_steps = int(len(train_examples) / opt.train_batch_size / opt.gradient_accumulation_steps * opt.num_train_epochs)
      
    # Prepare tokenizer
    #tokenizer = torch.load(opt.bert_tokenizer)
    tokenizer = RobertaTokenizer.from_pretrained(opt.bert_tokenizer)
        
    # Prepare model
    
    model = ini_from_pretrained(opt)
    
    model_config = model.config
    model = nn.DataParallel(model, device_ids=gpu_ls)
    model.config = model_config
    model.cuda(gpu_ls[0])
    
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
    all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
    all_atten_masks = torch.tensor(select_field(train_features, 'atten_masks'), dtype=torch.long)
    
    all_answers = torch.tensor([f.answer for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_example_ids, all_input_ids, all_segment_ids,all_atten_masks, all_answers)
    if opt.local_rank == -1:
      train_sampler = RandomSampler(train_data)
    else:
      train_sampler = DistributedSampler(train_data)
    # train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=opt.train_batch_size)
    
    loss_nsp_fn = torch.nn.CrossEntropyLoss()
    
    best_eval_acc=0.0
    best_test_acc=0.0
    best_step=0
    eval_acc_list=[]
    name = parse_opt_to_name(opt)
    
    name_plus = opt.train_data_dir.replace("/users4/ldu/abductive/data/sentences_order_train", "")
    name_plus = name_plus.replace('.pkl',  '')
    name = name + name_plus
    print(name)
    time_start = str(int(time.time()))[-6:]
    
    test_examples_all = load_examples(os.path.join(opt.test_data_dir))
    test_features_all = convert_examples_to_features(test_examples_all, tokenizer, opt.max_seq_length, True)
    
    #torch.cuda.set_device(opt.gpuls)
    
    for epoch in range(opt.num_train_epochs):
      print("Epoch:",epoch)
      tr_loss = 0
      nb_tr_examples, nb_tr_steps = 0, 0
      # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
      #freeze_params(model, requires_grad=True)
      #if epoch < opt.num_frozen_epochs:
      #    freeze_params(model, requires_grad=False)
     
      for step, batch in enumerate(train_dataloader):
          model.train()
          batch = tuple(t.cuda(gpu_ls[0]) for t in batch)
    
          example_ids, input_ids, segment_ids, atten_masks, answers = batch
          num_choices = input_ids.shape[1]
          
          accurancy = None
          
          input_ids_tmp = input_ids.squeeze()
          answers_tmp = answers
          segment_ids_tmp = segment_ids.squeeze()
          atten_masks_tmp = atten_masks.squeeze()
          model = model.train()
          
          #cls_scores = model(input_ids = input_ids_tmp, token_type_ids = segment_ids_tmp, attention_mask=atten_masks_tmp)
          cls_scores = model(input_ids = input_ids_tmp, attention_mask=atten_masks_tmp)
          
          cls_scores = cls_scores[0].softmax(-1)
          loss = loss_nsp_fn(cls_scores, answers_tmp)
          
          num_correct = float((answers_tmp==cls_scores.max(1)[1]).sum().detach().cpu())
          accuracy = num_correct / answers_tmp.shape[0]
          
          f = open(wkdir + name + '_' + time_start + '.csv', 'a+')
          f.write(str(loss.detach().cpu().numpy()) + ',' + str(accuracy) + '\n')
          f.close()
          
          if step % 20 == 0:
              print("step:", step, "loss:", loss.detach().cpu().numpy())
    
          if opt.fp16 and opt.loss_scale != 1.0:
              loss = loss * opt.loss_scale
          if opt.gradient_accumulation_steps > 1:
              loss = loss / opt.gradient_accumulation_steps
          loss.backward()
          tr_loss += loss.item()
          nb_tr_examples += input_ids.size(0)
          nb_tr_steps += 1
          if (step + 1) % opt.gradient_accumulation_steps == 0:
              optimizer.step()
              model.zero_grad()
              global_step += 1
                  
          ls = [model.config, model.state_dict()]
          
          if (step * opt.train_batch_size) % 192000 == 0:
              torch.save(ls, "/users4/ldu/abductive/pretrained_models/event_order_pretrained/" + str(step) + str(accuracy) + "e_" + str(epoch) + name + time_start + '.pkl')
          

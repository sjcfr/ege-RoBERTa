import pickle
import json
import numpy as np
import pandas as pd
import random

import torch
from onmt.DataSet import *
from onmt.BertModules import *
from onmt.GraphBert import *
from onmt.GraphTransformer import *
from onmt.VariationalGraphBert import *

import re
import copy
import pdb

class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj
    def read(self, size):
        return self.fileobj.read(size).encode()
    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()


def load_examples(input_file):
    f = open(input_file, 'rb')
    try:
        examples = pickle.load(f)
    except:
        f.close()
        f = open(input_file, 'r')
        examples = pickle.load(StrToBytes(f), encoding='iso-8859-1')
    f.close()
    return examples
    

def parse_gpuid(gpuls):
    ls = [int(n) for n in str(gpuls)]
    return ls
    
    
def parse_opt_to_name(opt):
    if 'large' in opt.bert_model or '_l_' in opt.bert_model:
        model_size = 'l'
    else:
        model_size = 'b'
    
    model_type = opt.model_type
    
    lr = opt.learning_rate
    warmup_proportion = opt.warmup_proportion
    margin = opt.do_margin_loss * opt.margin
    sep_sent = str(opt.sep_sent)[0]
    layer_norm = str(opt.layer_norm)[0]    
    
    if opt.model_type != 'pb':
        start_layer = opt.start_layer
        merge_layer = opt.merge_layer
        n_layer_extractor = opt.n_layer_extractor
        n_layer_aa = opt.n_layer_aa
        n_layer_gnn = opt.n_layer_gnn
        if opt.method_gnn == 'skip':
            n_layer_gnn = 0
        n_layer_merger = opt.n_layer_merger
        method_extractor = opt.method_extractor[0]
        method_merger = opt.method_merger[0]
        smooth_term = opt.loss_aa_smooth
        smooth_method = opt.loss_aa_smooth_method[0]
        pretrain_method = opt.pretrain_method
        if pretrain_method != 'R':
            pretrain_number = 'a'
        else:
            pretrain_number = str(int(opt.pretrain_number/10000))
        #lr = opt.learning_rate
        #warmup_proportion = opt.warmup_proportion
        #margin = opt.do_margin_loss * opt.margin
        Lambda = opt.Lambda_kl
        #sep_sent = str(opt.sep_sent)[0]
        #layer_norm = str(opt.layer_norm)[0]
        if 'match' in opt.test_data_dir:
            test_type = 'm'
        else:
            test_type = 'um' 
        name = [str(n) for n in [model_type, model_size, pretrain_method, pretrain_number, start_layer, merge_layer, n_layer_extractor, n_layer_aa, n_layer_gnn, n_layer_merger, smooth_term, smooth_method, Lambda, sep_sent, layer_norm, method_extractor, method_merger, lr, warmup_proportion, margin, test_type]]
    else:
        name = [str(n) for n in [model_type, model_size, pretrain_method, sep_sent, layer_norm, lr, warmup_proportion, margin]]
    
    name = "_".join(name)

    return name
    

def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training, 
                                 baseline=False, voc=None):
    """Loads a data file into a list of `InputBatch`s."""

    # roc is a multiple choice task. To perform this task using Bert,
    # we will use the formatting proposed in "Improving Language
    # Understanding by Generative Pre-Training" and suggested by
    # @jacobdevlin-google in this issue
    # https://github.com/google-research/bert/issues/38.
    #
    # Each choice will correspond to a sample on which we run the
    # inference. For a given roc example, we will create the 4
    # following inputs:
    # - [CLS] obs [SEP] choice_1 [SEP]
    # - [CLS] obs [SEP] choice_2 [SEP]
    # - [CLS] obs [SEP] choice_3 [SEP]
    # - [CLS] obs [SEP] choice_4 [SEP]
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.
    features = []
    if 'graph' in examples[0].keys():
        has_graph = True
    else:
        has_graph = False
    num_not_append = 0
    for example_index, example in enumerate(examples):
        
        obs_sentences = example['obs']
        hyps = example['hyps']
        hyps_chain = [copy.deepcopy(obs_sentences), copy.deepcopy(obs_sentences)]
        
        for hyp_chain, hyp in zip(hyps_chain, hyps):
            hyp_chain.insert(1, hyp)
        
        choices_features = []
        
        if_append = True
        
        for hyp_index, hyp_chain in enumerate(hyps_chain):
            
            chain_tokens_tmp = []
            sentence_ind_tmp = []
        
            for ith_sent, sent in enumerate(hyp_chain):
                
                sent_tokens = tokenizer.tokenize(sent)
                chain_tokens_tmp.append(sent_tokens)
                #obs_tokens = obs_tokens + ['.']
    
                sentence_ind_tmp.extend([ith_sent] * (len(sent_tokens) + 1))                

            tokens_tmp = ["<s>"] + chain_tokens_tmp[0] + ["</s>"] + chain_tokens_tmp[1] + ["</s>"] + chain_tokens_tmp[2] + ["</s>"]
                                                
            if baseline:
                sentence_ids = sentence2ids(hyp_chain, voc)
            
            input_ids_tmp = tokenizer.convert_tokens_to_ids(tokens_tmp)
            input_mask_tmp = [1] * len(input_ids_tmp)

            # Zero-pad up to the sequence length.
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
                graph = example['graph'][hyp_index]
                #graph = np.ones([3, 3]) * 0.33
            else:
                graph = None
            
            try:
                assert len(input_ids_tmp) == max_seq_length
                assert len(input_mask_tmp) == max_seq_length
                assert len(sentence_ind_tmp) == max_seq_length
            except:
                pdb.set_trace()
            
            if set(sentence_ind_tmp) != {0, 1, 2} and set(sentence_ind_tmp) != {0, 1, 2, -1}: 
                if_append = False
                num_not_append += 1 
                print(num_not_append)
                #print(obs_sentences)
                print("Too long example, id:", example_index)
            
            if not baseline:
                choices_features.append((tokens_tmp, input_ids_tmp, input_mask_tmp, sentence_ind_tmp, graph))
            else:
                choices_features.append((tokens, input_ids_tmp, input_mask_tmp, sentence_ind_tmp, graph, sentence_ids))

        answer = [0] * len(example['hyps'])
        try:
            answer[int(example['ans'])-1] = 1
        except:
            pdb.set_trace()
            answer[example['answer']] = 1
        
        if if_append:
            features.append(
                InputFeatures(
                    example_id = example_index,
                    choices_features = choices_features,
                    answer = answer
                )
            )

    return features    
    
    
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
            
            
def ini_from_pretrained(config, graph_embedder=None):
    
    pretrained_bert = torch.load(config.bert_model)
    
    try:
        state_dict = pretrained_bert[1]
        bert_config = pretrained_bert[0] 
    except:
        state_dict = pretrained_bert.state_dict()
        bert_config = pretrained_bert.config
    
    graph_bert_config = bert_config    
    for k in dir(config):
        if "__" not in k:
            try:
                #getattr(graph_bert_config, k)
                if getattr(graph_bert_config, k) != getattr(config, k):
                    setattr(graph_bert_config, k, getattr(config, k))
            except:
                #if getattr(graph_bert_config, k) != getattr(config, k):
                setattr(graph_bert_config, k, getattr(config, k))
    graph_bert_config.is_pretrain = config.is_pretrain
    model_config = BertConfig(graph_bert_config)
    
    if config.model_type == 'gb':    
        graph_bert_model = GraphBertModel(model_config)
    elif config.model_type == 'pb':
        graph_bert_model = BertForNextSentencePrediction(model_config)
    elif config.model_type in ['vgb', 'vgb_c']:
        graph_bert_model = VariationalGraphBertModel(model_config)
    elif config.model_type == 'bt':
        graph_bert_model = BertForNextSentencePrediction(model_config)
    elif config.model_type in ['er', 'gt']: 
        graph_bert_model = GraphTransformerModel(model_config, graph_embedder)
    
    old_keys = []
    new_keys = []
        
    for key in state_dict.keys():
        new_key = key
        if config.model_type == 'pb':
            if 'module.' in new_key:
                new_key = new_key.replace('module.', '')
            elif 'bert.encoder.layer' in new_key:
                new_key = new_key.replace('bert.encoder.layer', 'encoder.bert_layers')

        elif config.model_type == 'gb':
            '''
            if 'bert_layers' in key:
                new_key = new_key.replace('bert_layers', 'layers')
            '''
            if 'module.bert.embeddings' in new_key:
                new_key = new_key.replace('module.bert.embeddings', 'embeddings')
            if 'bert.encoder.layer' in new_key:
                new_key = new_key.replace('bert.encoder.layer', 'encoder.bert_layers')
            if 'module.' in new_key:
                new_key = new_key.replace('module.', '')
            elif '.layer.' in key:
                new_key = new_key.replace('.layer.', '.bert_layers.')
        elif config.model_type == 'vgb':
            
            if config.is_pretrain:
                if 'encoder.layer' in new_key:
                    dot_place = new_key.replace('encoder.layer.','').find('.')
                    ith_layer = int(new_key.replace('encoder.layer.','')[:dot_place])

                    if ith_layer < config.merge_layer:
                        new_key_rep = new_key.replace('encoder.layer', 'encoder.bert_layers')
                    else:
                        new_key_rep = new_key.replace('encoder.layer', 'decoder.bert_layers')
                        new_key_rep = new_key_rep.replace(str(ith_layer), str(ith_layer - config.merge_layer))
                    new_key = new_key_rep
            else:
                if 'module.' in new_key:
                    new_key = new_key.replace('module.', '')
        elif config.model_type == 'vgb_c':
            
            if config.is_pretrain:
                if 'encoder.layer' in new_key:
                    dot_place = new_key.replace('encoder.layer.','').find('.')
                    ith_layer = int(new_key.replace('encoder.layer.','')[:dot_place])

                    if ith_layer < config.merge_layer:
                        new_key_rep = new_key.replace('encoder.layer', 'encoder.bert_layers')
                    else:
                        new_key_rep = new_key.replace('encoder.layer', 'decoder.bert_layers')
                        new_key_rep = new_key_rep.replace(str(ith_layer), str(ith_layer - config.merge_layer))
                    new_key = new_key_rep
            else:
                if 'module.' in new_key:
                    new_key = new_key.replace('module.', '')

        elif config.model_type == 'bt':
            #pdb.set_trace()
            if 'embeddings.' in new_key:
                new_key = new_key.replace('embeddings.', 'bert.embeddings.', 1)

            if 'encoder.layer' in new_key:
                new_key = new_key.replace('encoder.layer', 'bert.encoder.layer')
            
        if new_key != key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)
        
    for name, parameter in graph_bert_model.state_dict().items():
        
        if name in state_dict.keys():
            try:
                bert_p = state_dict[name]
                parameter.data.copy_(bert_p.data)
            except:
                print('dimension mismatch! ' + name)
        else:
            print(name)
            
    graph_bert_model.keys_bert_parameter = state_dict.keys()
    
    return graph_bert_model


def freeze_params(model, requires_grad=False):
    freeze_ls = ['graph_extractor',
                   'adjacancy_approximator',
                   'gnn',
                   'merger_layers']
    for module in freeze_ls:
        
        parameters = getattr(model.encoder, module) 
        for param in parameters.parameters():
            param.requires_grad = requires_grad


def accuracy(out, labels):
    pdb.set_trace()
    out = np.array(out)
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]
    
    
def loss_graph(appro_matrix, true_graph, loss_fn, smooth_term=0,  method='all'):
    assert appro_matrix.shape == true_graph.shape
    L = appro_matrix.shape[1]
    loss_tot = 0
    for i in range(L):
        
        for j in range(L):
            if method ==  'all':

                p = torch.distributions.binomial.Binomial(1, appro_matrix[:,i, j].unsqueeze(-1))
                q = torch.distributions.binomial.Binomial(1, true_graph[:,i, j].unsqueeze(-1))
                #pdb.set_trace()
                ##p = torch.distributions.categorical.Categorical(appro_matrix[:,i,:])
                ##q = torch.distributions.categorical.Categorical(true_graph[:,i,:])
            else:
                x = appro_matrix[:,i,:]
                y = true_graph[:,i,:]
                x[:, i] += 1
                y[:, i] += 1
                
                p = torch.distributions.categorical.Categorical(x)
                q = torch.distributions.categorical.Categorical(y)
            
            loss_tmp = loss_fn(q, p).sum()
            loss_tot = loss_tot + loss_tmp
        
        '''
        p = torch.distributions.categorical.Categorical(appro_matrix[:,i,:] + smooth_term)
        q = torch.distributions.categorical.Categorical(true_graph[:,i,:] + smooth_term)
        loss_tmp = loss_fn(q, p).sum()
        loss_tot = loss_tot + loss_tmp
        '''
    return loss_tot
    
    
def write_result_to_file(args,result):
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        writer.write(result+"\n")

def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)

def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if param_model.grad is not None:
            if test_nan and torch.isnan(param_model.grad).sum() > 0:
                is_nan = True
            if param_opti.grad is None:
                param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan


def do_evaluation(model,eval_dataloader, opt, gpu_ls=None,  output_res=False):

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    logits_all = None
    if gpu_ls:
        gpu_id = gpu_ls[0]
    else:
        gpu_id = opt.gpuid

    with torch.no_grad():
    
        #res = {'ids':[],'pred':[],'ans':[]}
        res = pd.DataFrame()
        preds = []
        for dat in eval_dataloader:
            dat = [t.cuda(gpu_id) for t in dat]
            if len(dat) == 5 and not opt.baseline:
                input_ids, input_masks, sentence_inds, graphs, answers = dat
            elif len(dat) == 5 and opt.baseline:
                input_ids, input_masks, sentence_inds, answers, graph_ids = dat
                graphs = None
            else:
                input_ids, input_masks, sentence_inds, answers = dat
                graphs = None
            
            answers = answers 
            num_choices = input_ids.shape[1]
            cls_score = []
            res_tmp = []
            for n in range(num_choices):
               
                input_ids_tmp = input_ids[:,n,:]
                input_masks_tmp = input_masks[:,n,:]
                sentence_inds_tmp = sentence_inds[:,n,:]
                answers_tmp = answers[:,n]
                
                if opt.baseline:
                    graph_ids_tmp = graph_ids[:,n,:]
                
                if graphs is not None:
                    graphs_tmp = graphs[:,n,:]
                else:
                    graphs_tmp = graphs                    
                graphs_tmp_scaled = graphs_tmp
                
                if opt.model_type == 'gb':  
                    _, cls_score_tmp, attn_scores = model(input_ids = input_ids_tmp, 
                                                      sentence_inds = sentence_inds_tmp, 
                                                      graphs = graphs_tmp_scaled) ##
                elif opt.model_type in ['er', 'gt']: 
                    _, cls_score_tmp = model(input_ids = input_ids_tmp, 
                                                      graph_ids = graph_ids_tmp,
                                                      sentence_inds = sentence_inds_tmp, 
                                                      graphs = graphs_tmp_scaled) ##
                elif opt.model_type in ['pb',  'bt']:
                    cls_score_tmp = model(input_ids = input_ids_tmp) ##
                elif opt.model_type in ['vgb', 'vgb_c']:
                    cls_score_tmp, attn_scores = model(input_ids_p = input_ids_tmp, 
                                                        attn_mask_p = input_masks_tmp,
                                                          sentence_inds_p = sentence_inds_tmp) ##                                 
                
                cls_score_tmp = cls_score_tmp.softmax(-1)
                cls_score.append(cls_score_tmp.detach().cpu().numpy()[:,1].tolist())
                
                res_tmp.append(cls_score_tmp.detach().cpu().numpy()[:,0].tolist())
                res_tmp.append(cls_score_tmp.detach().cpu().numpy()[:,1].tolist())
                
            #pdb.set_trace()
            cls_score = np.array(cls_score).T
            answers = answers.detach().cpu().numpy()
            num_acc_tmp = sum(cls_score.argmax(1) == answers.argmax(1))
            preds.extend(cls_score.argmax(1).tolist())
            
            res_tmp.append(answers.argmax(1))
            res_tmp = pd.DataFrame(np.array(res_tmp).T)
            res = res.append(res_tmp)
            #print(num_acc_tmp)
            #eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += num_acc_tmp

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

    #eval_loss = eval_loss / nb_eval_steps
    
    eval_accuracy = float(eval_accuracy) / nb_eval_examples

    model.zero_grad()
    '''
    if not output_res:
        return eval_accuracy
    else:
        return eval_accuracy, preds  
    '''
    return eval_accuracy, preds

def graph_ids_to_tensor(all_graph_ids, opt):
    if 'mcnc' in opt.train_data_dir:
        max_L = 7
    elif 'roc' in opt.train_data_dir:
        max_L = 15
        
    PAD_token = 2
    all_graph_ids_padded = []
    for sample in all_graph_ids:
        all_graph_ids_padded.append([])
        for candidate in sample:
            all_graph_ids_padded[-1].append([])
            for sentence in candidate:
                l_sent = len(sentence[0])
                if l_sent > max_L:
                    sentence[0] = sentence[0][:max_L]
                elif l_sent < max_L:
                    l_diff = max_L - l_sent
                    pad_ls = [PAD_token] * l_diff
                    
                    sentence[0] = sentence[0] + pad_ls
                    #sentence = torch.LongTensor(sentence)
                all_graph_ids_padded[-1][-1].append(sentence)
             
    all_graph_ids_padded = torch.LongTensor(all_graph_ids_padded)
    all_graph_ids_padded = all_graph_ids_padded.squeeze()
    
    return all_graph_ids_padded
    
    
def retro(key, graph):
    key_expand = [key + "_obj", key + '_subj']
    res = []
    for key_tmp in key_expand:
        try:
            fwd_nodes_tmp = graph[key_tmp]
            res.append(fwd_nodes_tmp)
        except:
            pass
            
    return res
  

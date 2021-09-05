import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from onmt.BertModules import *
from onmt.GraphBert import *

import pdb

class VariationalGraphExtractor(nn.Module):
    def __init__(self, config):
        super(VariationalGraphExtractor, self).__init__()
        
        self.num_layers = config.n_layer_extractor
        
        self.config = copy.deepcopy(config)
        self.config.method = config.method_extractor
        self.config.layer_norm = config.layer_norm
        
        layer = GATLayer(self.config)
        self.extract_layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(self.config.n_layer_extractor)])
        
    def forward(self, sent_ind, start_layer, subsequent_layers=None):
        '''
        start_layer: tensor with shape batch * seq_length * dim
                       seq_length = number of tokens
        subsequent_layers: list, each element is a tensor with shape batch * seq_length * dim
        graph_vectors: tensor with shape batch * seq_length * dim
                       seq_length = number of sentences
        '''
        
        assert len(subsequent_layers) == self.num_layers
        
        batch = sent_ind.shape[0]
        num_sent = sent_ind.max().detach().cpu().numpy() + 1
        
        graph_vectors = []
        for n in range(batch):
            graph_vectors_sample = []
            for ith_sent in range(num_sent):
                if ith_sent == 0:
                    graph_vectors_sample_ith = start_layer[n][sent_ind[n] == ith_sent][0].unsqueeze(0)
                    graph_vectors_sample.append(graph_vectors_sample_ith)
                    
                    graph_vectors_sample_ith = start_layer[n][sent_ind[n] == ith_sent][1:].mean(0).unsqueeze(0)
                    graph_vectors_sample.append(graph_vectors_sample_ith)
                else:
                    graph_vectors_sample_ith = start_layer[n][sent_ind[n] == ith_sent].mean(0).unsqueeze(0)
                    graph_vectors_sample.append(graph_vectors_sample_ith)
                    
            try:
                graph_vectors_sample = torch.cat(graph_vectors_sample).unsqueeze(0)
            except:
                pdb.set_trace()
            graph_vectors.append(graph_vectors_sample)
        graph_vectors = torch.cat(graph_vectors)
        
        for i in range(self.num_layers):
            #graph_vectors = self.extract_layers[i](graph_vectors, subsequent_layers[i],  sent_ind=sent_ind)
            graph_vectors = self.extract_layers[i](graph_vectors, subsequent_layers[i])
        
        return graph_vectors
                                                          
        
class TransformerEncoder(nn.Module):
    def __init__(self, config, keep_multihead_output=False):
        super(TransformerEncoder, self).__init__()

        self.config = config
        
        bert_layer = BertLayer(config, output_attentions=False,
                                  keep_multihead_output=keep_multihead_output)
                                  
        self.bert_layers = nn.ModuleList([copy.deepcopy(bert_layer) for _ in range(config.merge_layer)])
        self.graph_extractor = VariationalGraphExtractor(config)            
        
    def forward(self, hidden_states, attention_mask, sentence_ind, true_adjacancy_matrix=None, output_all_encoded_layers=True, head_mask=None):
        all_encoder_layers = []
        all_attentions = []
        
        merge_layer = self.config.merge_layer
        start_layer = self.config.start_layer
        num_sub_layers = self.config.n_layer_extractor
        
        assert start_layer + num_sub_layers <= merge_layer
        
        def append(hidden_states):
            all_encoder_layers.append(hidden_states)
                        
        for i in range(merge_layer):
            hidden_states = self.bert_layers[i](hidden_states, attention_mask, head_mask[i])
            append(hidden_states)
        
        context_vector_start = all_encoder_layers[start_layer]
        context_vector_subsequent = all_encoder_layers[(start_layer + 1): (start_layer + num_sub_layers + 1)]
        
        graph_vectors = self.graph_extractor(sentence_ind, context_vector_start, context_vector_subsequent)
        
        return all_encoder_layers, graph_vectors
        
        
        
class TransformerDecoder(nn.Module):
    def __init__(self, config, keep_multihead_output=False):
        super(TransformerDecoder, self).__init__()

        self.config = config
        
        bert_layer = BertLayer(config, output_attentions=False,
                                  keep_multihead_output=keep_multihead_output)
                                  
        self.num_bert_layers = (config.num_hidden_layers - config.merge_layer) 
        self.bert_layers = nn.ModuleList([copy.deepcopy(bert_layer) for _ in range(self.num_bert_layers)])
                
        self.gnn = GNN(config)
        
        if config.method_merger == 'gat':
            merger_layer = GATResMergerLayer(config)
            self.merger_layers = nn.ModuleList([merger_layer for _ in range(config.n_layer_merger)])
        elif config.method_merger == 'add':
            merger_layer = AddMergerLayer(config)
            self.merger_layers = nn.ModuleList([merger_layer for _ in range(config.n_layer_merger)])
        else:
            raise NotImplementedError        
        
    def forward(self, hidden_states, graph_vectors, attn_scores=None, true_adjacancy_matrix=None, attention_mask=None, output_all_encoded_layers=True, head_mask=None):
        all_encoder_layers = []
        all_attentions = []
        
        num_merger_layers = self.config.n_layer_merger
                
        def append(hidden_states):
            all_encoder_layers.append(hidden_states)
            
        graph_vectors = self.gnn(attn_scores, graph_vectors) 
                
        for ith_merger_layer in range(num_merger_layers):
            ith_bert_layer = ith_merger_layer
            if self.config.method_merger != 'combine':
                hidden_states = self.merger_layers[ith_merger_layer](hidden_states, graph_vectors, sent_ind=None)
                hidden_states = self.bert_layers[ith_bert_layer](hidden_states, attention_mask=None, head_mask=None)
            else:
                attention_mask_tmp = torch.cat([attention_mask, attention_mask[:, :, :, :(graph_vectors.shape[1] + 0)] * 0], -1) 
                hidden_states = torch.cat([hidden_states, graph_vectors], 1)
                hidden_states = self.bert_layers[ith_bert_layer](hidden_states, attention_mask_tmp, head_mask[ith_bert_layer])
                hidden_states = hidden_states[:, :-(graph_vectors.shape[1] + 0), :]
                
            all_encoder_layers.append(hidden_states)
        
        if self.num_bert_layers > num_merger_layers:
            for j in range(num_merger_layers, self.num_bert_layers):
                hidden_states = self.bert_layers[j](hidden_states, attention_mask=None, head_mask=None)
                all_encoder_layers.append(hidden_states)
        
        return all_encoder_layers, attn_scores
        

class Inferer(nn.Module):
    def __init__(self, config):
        super(Inferer, self).__init__()
        
        self.num_layers = config.n_layer_aa
        
        self.config = copy.deepcopy(config)
        self.config.method = 'self'
        self.config.layer_norm = config.layer_norm
        
        layer = GATLayer(self.config)
        layer.attn_drop = False
        
        self.aa_layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(self.num_layers)])
        self.final_layer = BertSelfAttention(config)
        self.final_layer.output_attentions = True
        self.final_layer.do_softmax = False
        self.final_layer.unmix = True
        self.final_layer.attn_drop = False
        
    def forward(self, graph_vectors):
        
        for i in range(self.num_layers):
            graph_vectors = self.aa_layers[i](graph_vectors)
        ''' 
        attn_scores, graph_vectors = self.final_layer(graph_vectors)
        attn_scores = attn_scores[:,:,1:,1:]
        z = graph_vectors[:,0,:]  
        graph_vectors = graph_vectors[:,1:,:]  
        '''
        z = graph_vectors[:,0,:]  
        graph_vectors = graph_vectors[:,1:,:]  
        attn_scores, _ = self.final_layer(graph_vectors)
        
        #attn_scores = (attn_scores / attn_scores.sum(-1).unsqueeze(1))    
        #attn_scores = nn.Softmax(dim=-1)(attn_scores)
        
        #pdb.set_trace()
        #attn_scores = (attn_scores / attn_scores.sum(2).unsqueeze(1))
        
        return attn_scores, z, graph_vectors


class PostInferer(nn.Module):
    def __init__(self, config):
        super(PostInferer, self).__init__()
        
        self.num_layers = config.n_layer_aa
        
        self.config = copy.deepcopy(config)
        self.config.method = 'self'
        self.config.layer_norm = config.layer_norm
        
        layer = GATLayer(self.config)
        layer.attn_drop = False
        
        self.aa_layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(self.num_layers + 1)])
        
    def forward(self, graph_vectors, graph):
    
        graph = graph.squeeze()
    
        relay_node = graph_vectors[:,0,:]  
        satellite = graph_vectors[:,1:,:]
        
        relay_node = self.aa_layers[0](relay_node.unsqueeze(1),  satellite)
        
        satellite = torch.matmul(satellite.transpose(1,2), graph).transpose(1,2)
        
        graph_vectors = torch.cat([relay_node, satellite],  dim=1)
        
        for i in range(1, self.num_layers):
            graph_vectors = self.aa_layers[i](graph_vectors)
            
        relay_node_u = graph_vectors[:,0,:]
        satellite_u = graph_vectors[:,1:,:]
            
        z = self.aa_layers[-1](relay_node_u.unsqueeze(1), satellite_u).squeeze()
        attn_scores = None
                
        return attn_scores, z, satellite_u
        
    
class VariationalGraphBertModel(BertPreTrainedModel):

    def __init__(self, config, output_attentions=False, keep_multihead_output=False):
        super(VariationalGraphBertModel, self).__init__(config)
        self.config = config
        self.is_pretrain = config.is_pretrain
        self.output_attentions = output_attentions
        
        self.embeddings = BertEmbeddings(config)
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        self.prior_inferer = Inferer(config)
        if self.is_pretrain:
            if self.config.model_type == 'vgb':
                self.post_inferer = Inferer(config)
            if self.config.model_type == 'vgb_c':
                self.post_inferer = PostInferer(config)
        
        self.pooler = BertPooler(config)
        self.cls = BertOnlyNSPHead(config)
        
        if self.is_pretrain:
            self.lm_head = BertLMPredictionHead(config, self.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_multihead_outputs(self):
        """ Gather all multi-head outputs.
            Return: list (layers) of multihead module outputs with gradients
        """
        return [layer.attention.self.multihead_output for layer in self.encoder.layer]
    ##
    def forward(self, input_ids_p, input_ids_q=None, attn_mask_p = None, sentence_inds_p=None, attn_mask_q=None, sentence_inds_q=None, token_type_ids=None, graph=None, 
                      is_train=True, output_all_encoded_layers=False, head_mask=None):
        if token_type_ids is None:
            token_type_ids_p = torch.zeros_like(input_ids_p)
            if self.is_pretrain:
                token_type_ids_q = torch.zeros_like(input_ids_q)

        '''
        if attention_mask is None:
            attention_mask_p = torch.ones_like(input_ids_p)
            attention_mask_q = torch.ones_like(input_ids_q)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        '''
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand_as(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers
        
        embedding_output_p = self.embeddings(input_ids_p, token_type_ids_p)
        if self.is_pretrain:
            embedding_output_q = self.embeddings(input_ids_q, token_type_ids_q)
        
        encoded_layers_p, graph_vectors_p = self.encoder(hidden_states = embedding_output_p, attention_mask = attn_mask_p,
                                                         sentence_ind = sentence_inds_p, 
                                                         output_all_encoded_layers=output_all_encoded_layers,
                                                         head_mask=head_mask)
        if self.is_pretrain:
            encoded_layers_q, graph_vectors_q = self.encoder(hidden_states = embedding_output_q, attention_mask = attn_mask_q,
                                                             sentence_ind = sentence_inds_q, 
                                                             output_all_encoded_layers=output_all_encoded_layers,
                                                             head_mask=head_mask)
                                                         
        attn_scores_p, z_p, graph_vectors_p = self.prior_inferer(graph_vectors_p)
        if self.is_pretrain:
            if self.config.model_type == 'vgb':
                attn_scores_q, z_q, graph_vectors_q = self.post_inferer(graph_vectors_q)
            elif self.config.model_type == 'vgb_c':
                attn_scores_q, z_q, graph_vectors_q = self.post_inferer(graph_vectors_q, graph)
        
        if self.is_pretrain:
            graph_vectors = torch.cat([z_q.unsqueeze(1), graph_vectors_q], axis=1)
        else:
            graph_vectors = torch.cat([z_p.unsqueeze(1), graph_vectors_p], axis=1)
        
        encoded_layers, attn_scores = self.decoder(encoded_layers_p[-1], graph_vectors, attention_mask=None, attn_scores=attn_scores_p)
        
        if self.output_attentions:
            all_attentions, encoded_layers, attn_scores = encoded_layers
        else:
            #encoded_layers, attn_scores = encoded_layers
            encoded_layers = encoded_layers[-1]
            
        attn_scores_p = attn_scores_p.squeeze(1)
        
        if self.is_pretrain:
            if self.config.model_type == 'vgb':
                attn_scores_q = attn_scores_q.squeeze(1)
            elif self.config.model_type == 'vgb_c':
                pass
        
        pooled_output = self.pooler(encoded_layers)
        cls_scores = self.cls(pooled_output)
        
        if self.is_pretrain:
            predictions_lm = self.lm_head(encoded_layers)
                
        if self.is_pretrain:
            return predictions_lm, z_p, z_q, attn_scores_q
        else:
            return cls_scores, attn_scores_p
    
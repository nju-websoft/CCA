import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .knowformer_encoder import Embeddings, Encoder, truncated_normal_init, norm_layer_init
import time


class Knowformer(nn.Module):
    def __init__(self, config):
        super(Knowformer, self).__init__()
        self._emb_size = config['hidden_size']
        self._n_layer = config['num_hidden_layers']
        self._n_head = config['num_attention_heads']
        self._input_dropout_prob = config['input_dropout_prob']
        self._attention_dropout_prob = config['attention_dropout_prob']
        self._hidden_dropout_prob = config['hidden_dropout_prob']
        self._residual_dropout_prob = config['residual_dropout_prob']
        self._context_dropout_prob = config['context_dropout_prob']
        self._initializer_range = config['initializer_range']
        self._intermediate_size = config['intermediate_size']

        
        self._voc_size = config['vocab_size']
        self._n_relation = config['num_relations']
        self.device = config['device']
        self.ele_embedding = Embeddings(self._emb_size, self._voc_size, self._initializer_range)

        self.triple_encoder = Encoder(config)
        self.context_encoder = Encoder(config)

        self.input_dropout_layer = nn.Dropout(p=self._input_dropout_prob)
        self.context_dropout_layer = nn.Dropout(p=self._context_dropout_prob)

    def __forward_triples(self, triple_ids, neighbor_ids, neighbors_trustworthy,  each_layer = False):
        # convert token id to embedding
        emb_out = self.ele_embedding(triple_ids)  # (batch_size, 4, embed_size)
        emb_out = self.input_dropout_layer(emb_out)
        # integrate neighbors ids
        if neighbor_ids!=None and len(neighbor_ids[0])>0:
            neighbor_out = self.ele_embedding(neighbor_ids)
            neighbor_out = self.input_dropout_layer(neighbor_out)        
            emb_out = torch.cat([emb_out, neighbor_out],1) 
        encoder = self.triple_encoder
        # mask is to cover some values in attention score
        emb_out = encoder(emb_out, neighbors_trustworthy, each_layer, mask = None)
        # emb_out = encoder(emb_out, mask=None)  # (batch_size, 4*(1+neighbor_num), embed_size)

        return emb_out

    def __process_mask_feat(self, mask_feat):
        return torch.matmul(mask_feat, self.ele_embedding.lut.weight.transpose(0, 1))

    def forward(self, src_ids, neighbor_ids, neighbors_trustworthy, kgc = False, head_mask = False, each_layer = False):
        # src_ids: (batch_size, seq_size, 1)
        # window_ids: (batch_size, seq_size) * neighbor_num
        if not kgc:
            # 1. do not use embeddings from neighbors
            seq_emb_out = self.__forward_triples(src_ids, neighbor_ids, neighbors_trustworthy, each_layer)
            # cls_emb = seq_emb_out[:, 0, :]  # (batch_size,4, embed_size)
            cls_emb = None
            cls_emb_with_neighbor = seq_emb_out[:,1:4, :]
            batch_num, word_num, embedding = cls_emb_with_neighbor.shape
            cls_emb_with_neighbor = torch.reshape(cls_emb_with_neighbor, (batch_num, word_num * embedding))
            result = cls_emb_with_neighbor
            logit = None
        else:
            seq_emb_out = self.__forward_triples(src_ids, neighbor_ids, neighbors_trustworthy, each_layer)
            # get head_mask or tail mask emb
            if not each_layer:
                if head_mask:
                    mask_emb = seq_emb_out[:,1,:]
                else:
                    mask_emb = seq_emb_out[:,3,:]
                logit = seq_emb_out[:, 1:4,:]
                logits_from_triplets = self.__process_mask_feat(mask_emb)  # (batch_size, vocab_size)
                result = logits_from_triplets
            else:
                result = []
                logit = []
                for i in range(len(seq_emb_out)):
                    if head_mask:
                        mask_emb = seq_emb_out[i][:,1,:]
                    else:
                        mask_emb = seq_emb_out[i][:,3,:]
                    logit.append(seq_emb_out[i][:, 1:4,:])
                    logits_from_triplets = self.__process_mask_feat(mask_emb)  # (batch_size, vocab_size)
                    result.append(logits_from_triplets)
        return result, logit
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adamw import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import get_cosine_schedule_with_warmup
from contiguous_params import ContiguousParams
from torch.cuda.amp import autocast
from .knowformer import Knowformer
from .utils import get_ranks, get_norms, get_scores


class NFormer(nn.Module):
    def __init__(self, args: dict, bert_encoder: Knowformer):
        super(NFormer, self).__init__()

        self.use_kgc = args['use_kgc']

        self.device = torch.device(args['device'])
        self.add_neighbors = True if args['add_neighbors'] else False
        self.neighbor_num = args['neighbor_num']
        self.lr = args['kge_lr']
        self.entity_begin_idx = args['struc_entity_begin_idx']
        self.entity_end_idx = args['struc_entity_end_idx']
        self.use_extra_encoder = args['extra_encoder']
        self.sigmoid = nn.Sigmoid()
        self.bert_encoder = bert_encoder
        self.mlp = nn.Sequential(
            nn.Linear(768,768),
            nn.ReLU(),
            nn.Linear(768,1)
        )
    
        self.bce_loss = nn.BCELoss(reduction = 'none')
        self.bce_loss_withlogits = nn.BCEWithLogitsLoss(reduction = 'none')
        # kgc loss_fc
        if args['low_degree']:
            args['kge_label_smoothing'] = 0.8
        else:
            args['kge_label_smoothing'] = 0
        self.loss_fc = nn.CrossEntropyLoss(label_smoothing=args['kge_label_smoothing'], reduction = 'none')
        self.score_loss_fc = nn.CrossEntropyLoss(reduction = 'none')
    def forward(self, batch_data, return_all_layer = False):
     
        if self.use_kgc:
            output = self.link_prediction(batch_data, return_all_layer)
        else:
            output = self.triple_classification(batch_data)
        return output['loss'], output['loss_info'], output['logits'], output['logits_score']

    def training_step(self, batch, return_all_layer = False):
        loss, loss_info, logits,logits_score = self.forward(batch, return_all_layer)
        return loss, loss_info, logits, logits_score

    def training_epoch_end(self, outputs):
        return np.round(np.mean([loss.item() for loss in outputs]), 4)


    def triple_classification(self, batch):
        # 1. prepare data
        input_ids = batch['struc_data']['input_ids'].to(self.device)  # batch_size * 4
        neighbors = batch['struc_neighbors']['input_ids'].to(self.device) # batch_size * neighbor_num * 3 
        neighbors_trustworthy = batch['neighbor_trustworthy']        
        if neighbors_trustworthy != None:
            neighbors_trustworthy = batch['neighbor_trustworthy'].to(self.device) 
        
        context_input_ids = None
        # the labels that used to ranking loss
        score_labels = batch['score_labels'].to(self.device)
        # the soft label that used to update
        train_labels = batch['soft_labels'].to(self.device)
        code = batch['code']
  
        # 2. encode
        origin_logits = self.bert_encoder(input_ids, neighbors, neighbors_trustworthy)
        logits = self.mlp(origin_logits)
        
        score_labels = torch.unsqueeze(score_labels,1).to(logits.dtype)
        train_labels = torch.unsqueeze(train_labels,1).to(logits.dtype)
        # get loss for backward
        train_loss = self.bce_loss_withlogits(logits, train_labels)
        # train_loss = self.bce_loss(logits, train_labels)
        batch_loss = train_loss.mean()
        # get loss for ranking
        
        score_loss = self.bce_loss_withlogits(logits, score_labels)
        score_loss = score_loss.squeeze()
        loss_info = [(c,l) for c,l in zip(code, score_loss.tolist())] 

        return {'loss': batch_loss,   'loss_info': loss_info, 'logits':origin_logits}
    
    def link_prediction(self, batch,  return_all_layer):
        
        head_input_ids = batch['struc_head_prompts']['input_ids'].to(self.device)  # batch_size * 3
        tail_input_ids = batch['struc_tail_prompts']['input_ids'].to(self.device)
        head_labels = batch['head_labels'].to(self.device)  # batch_size
        tail_labels = batch['tail_labels'].to(self.device)
        head_filters = batch['head_filters'].to(self.device)
        tail_filters = batch['tail_filters'].to(self.device)
        if batch['head_struc_neighbors']!= None:
            head_struc_neighbors = batch['head_struc_neighbors']['input_ids'].to(self.device)
            tail_struc_neighbors = batch['tail_struc_neighbors']['input_ids'].to(self.device)
            head_struc_neighbors_code = batch['head_struc_neighbors_code']
            tail_struc_neighbors_code = batch['tail_struc_neighbors_code']
        else:
            head_struc_neighbors, tail_struc_neighbors =None, None
        head_neighbors_trustworthy = batch['head_neighbor_trustworthy']        
        tail_neighbors_trustworthy = batch['tail_neighbor_trustworthy']
        if head_neighbors_trustworthy != None:
            head_neighbors_trustworthy = batch['head_neighbor_trustworthy'].to(self.device) 
            tail_neighbors_trustworthy = batch['tail_neighbor_trustworthy'].to(self.device) 
        code = batch['code']
        
        confidence = batch['soft_labels'].to(self.device)
        
        if not return_all_layer:
            head_logits, head_repre = self.bert_encoder(head_input_ids, head_struc_neighbors, head_neighbors_trustworthy,  kgc=True,head_mask =True, each_layer = return_all_layer)
            head_train_logits = head_logits
            # 3. compute loss and rank
            head_loss = self.loss_fc(head_train_logits, head_labels + self.entity_begin_idx)
                
            # head_logits = head_train_logits[:, self.entity_begin_idx: self.entity_end_idx]
            # rank = get_ranks(F.softmax(head_logits, dim=-1), head_labels, head_filters)
            tail_logits , tail_repre= self.bert_encoder(tail_input_ids, tail_struc_neighbors, tail_neighbors_trustworthy,  kgc=True, head_mask=False, each_layer = return_all_layer)
            tail_train_logits = tail_logits
            
            tail_loss = self.loss_fc(tail_train_logits, tail_labels + self.entity_begin_idx)
            loss =  confidence * (head_loss + tail_loss)
            train_loss = loss.mean()

            tail_logits_score = torch.softmax(tail_logits, dim = -1)[list(range(len(code))), tail_labels + self.entity_begin_idx]
            head_logits_score = torch.softmax(head_logits, dim = -1)[list(range(len(code))), head_labels + self.entity_begin_idx]
            logits_score = head_logits_score + tail_logits_score
            
            head_score = self.score_loss_fc(head_train_logits, head_labels+self.entity_begin_idx)
            tail_score = self.score_loss_fc(tail_train_logits, tail_labels + self.entity_begin_idx)
            score =  head_score + tail_score
            loss_info = [(c,l) for c,l in zip(code, score.tolist())] 
            logits_score = [(c,l) for c,l in zip(code, logits_score.tolist())] 
            repre = [head_repre , tail_repre]
            return {'loss': train_loss, 'loss_info': loss_info, 'logits': repre, 'logits_score': logits_score}
        else:
            head_logits, head_repre = self.bert_encoder(head_input_ids, head_struc_neighbors, head_neighbors_trustworthy,  kgc=True,head_mask =True, each_layer = return_all_layer)
            head_train_logits = head_logits[-1]
            # 3. compute loss and rank
            head_loss = self.loss_fc(head_train_logits, head_labels + self.entity_begin_idx)
                
            # head_logits = head_train_logits[:, self.entity_begin_idx: self.entity_end_idx]
            # rank = get_ranks(F.softmax(head_logits, dim=-1), head_labels, head_filters)
            tail_logits , tail_repre= self.bert_encoder(tail_input_ids, tail_struc_neighbors, tail_neighbors_trustworthy,  kgc=True, head_mask=False, each_layer = return_all_layer)
            tail_train_logits = tail_logits[-1]
            tail_loss = self.loss_fc(tail_train_logits, tail_labels + self.entity_begin_idx)
            repre = head_repre[-1] + tail_repre[-1]
            train_loss = 0
            loss_info = []
            for layer in range(len(head_logits)):
                score_logits = head_logits[layer]
                head_score_loss = self.loss_fc(score_logits, head_labels + self.entity_begin_idx)
                score_logits = tail_logits[layer]
                tail_score_loss = self.loss_fc(score_logits, tail_labels + self.entity_begin_idx)
                layer_score_loss = head_score_loss + tail_score_loss
                train_loss += layer_score_loss.mean()/3
                loss_info.append([(c,l) for c,l in zip(code, layer_score_loss.tolist())])          
            return {'loss': train_loss, 'loss_info': loss_info, 'logits': repre, 'logits_score': logits_score}


    ####################################################################################################################
    def configure_optimizers(self, total_steps: int):
        opt = torch.optim.AdamW(ContiguousParams(self.bert_encoder.parameters()).contiguous(), lr=self.lr)
        return {'optimizer': opt, 'scheduler': None}

    def get_parameters(self):
        decay_param = []
        no_decay_param = []
        mlp_param = []
        for n, p in self.bert_encoder.named_parameters():
            if not p.requires_grad:
                continue
            if ('bias' in n) or ('LayerNorm.weight' in n):
                no_decay_param.append(p)
            else:
                decay_param.append(p)
        
        for n, p in self.mlp.named_parameters():
            mlp_param.append(p)

        return [
            {'params': decay_param, 'weight_decay': 1e-2, 'lr': self.lr},
            {'params': no_decay_param, 'weight_decay': 0, 'lr': self.lr},
            {'params': mlp_param, 'weight_decay': 0, 'lr': self.lr}
        ]

    def freeze(self):
        for n, p in self.bert_encoder.named_parameters():
            p.requires_grad = False

    def clip_grad_norm(self):
        
        norms = get_norms(self.bert_encoder.parameters()).item()
        info = f'grads for N-Former: {round(norms, 4)}'
       
        return info

    def grad_norm(self):
        norms = get_norms(self.bert_encoder.parameters()).item()
        return round(norms, 4)
    



    
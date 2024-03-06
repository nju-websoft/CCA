import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from transformers import BertTokenizer, BertForMaskedLM, get_cosine_schedule_with_warmup
from torch.optim.adamw import AdamW
from torch.cuda.amp import autocast
from .utils import get_ranks, get_norms, get_scores


class NBert(nn.Module):
    def __init__(self, args: dict, tokenizer: BertTokenizer, bert: BertForMaskedLM):
        super(NBert, self).__init__()

        # 1. hyper params
        self.device = torch.device(args['device'])
        self.pretraining = True if args['task'] == 'pretrain' else False
        self.add_neighbors = True if args['add_neighbors'] else False
        self.neighbor_num = args['neighbor_num']
        self.neighbor_token = args['neighbor_token']
        self.lr = args['lm_lr']
        self.entity_begin_idx = args['text_entity_begin_idx']
        self.entity_end_idx = args['text_entity_end_idx']
        self.scheme  = args['scheme']
        # 2. model
        self.tokenizer = tokenizer
        self.bert_encoder = bert
        self.bert_encoder.resize_token_embeddings(len(self.tokenizer))
        
        self.mlp = nn.Sequential(
            nn.Linear(768,768),
            nn.ReLU(),
            nn.Linear(768,1)
        )
        self.bce_loss = nn.BCELoss(reduction = 'none')
        self.bce_loss_withlogits = nn.BCEWithLogitsLoss(reduction = 'none')
        self.sigmoid = nn.Sigmoid()
        if args['low_degree']:
            args['lm_label_smoothing'] = 0.8
        else:
            args['lm_label_smoothing'] = 0
        self.loss_fc = nn.CrossEntropyLoss(label_smoothing=args['lm_label_smoothing'])
        self.loss_fc_each = nn.CrossEntropyLoss(label_smoothing=args['lm_label_smoothing'], reduction = 'none')
        self.score_loss_fc_each = nn.CrossEntropyLoss(reduction ='none')
    def forward(self, batch_data):
        if self.pretraining:
            output = self.pretrain(batch_data)
            return output['loss'], output['loss_info'], None, output['rank']
        else:
            # if self.scheme == 'mlp':
            #     output = self.triple_classification(batch_data)
            output = self.link_prediction(batch_data)
            return output['loss'],  output['loss_info'], output['logits'], output['logits_score']

    def training_step(self, batch, batch_idx):
        loss, loss_info, logits,logits_score = self.forward(batch)
        return loss, loss_info, logits, logits_score

    def training_epoch_end(self, outputs):
        loss, rank = [], []
        for batch_loss, batch_rank in outputs:
            loss.append(batch_loss)
            rank += batch_rank
        scores = get_scores(rank, loss)
        return np.round(np.mean(loss), 2), scores
        # return np.round(np.mean([loss[0].item() for loss in outputs]), 2)

    def validation_step(self, batch, batch_idx):
        loss, rank ,sample_loss= self.forward(batch)
        return loss.item(), rank

    def validation_epoch_end(self, outputs):
        loss, rank = list(), list()
        for batch_loss, batch_rank in outputs:
            loss.append(batch_loss)
            rank += batch_rank
        loss = np.mean(loss)
        scores = get_scores(rank, loss)
        return scores

    def pretrain(self, batch):
        # 1. prepare data
        text_data = batch['text_data']
        input_ids = text_data['input_ids'].to(self.device)
        token_type_ids = text_data['token_type_ids'].to(self.device)
        attention_mask = text_data['attention_mask'].to(self.device)
        mask_pos = text_data['mask_pos'].to(self.device)
        labels = batch['labels'].to(self.device)
        filters = None

        # 2 encode with bert
        output = self.bert_encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        # 3. get logits, calculate loss and rank
        logits = output.logits[mask_pos[:, 0], mask_pos[:, 1], self.entity_begin_idx:self.entity_end_idx]
        loss = self.loss_fc(logits, labels)
        rank = get_ranks(F.softmax(logits, dim=-1), labels, filters)

        return {'loss': loss, 'rank': rank, 'logits': logits, 'loss_info': None}

    def encode_neighbors(self, batch_data):
        # 1. prepare data
        data = batch_data['text_neighbors']
        labels = batch_data['neighbors_labels'].to(self.device)  # (batch_size, )

        # 2. encode neighbors
        loss_from_neighbors = None
        embeds_from_neighbors = []
        for i in range(self.neighbor_num):
            input_ids = data[i]['input_ids'].to(self.device)  # (batch_size, seq_len)
            token_type_ids = data[i]['token_type_ids'].to(self.device)  # (batch_size, seq_len)
            attention_mask = data[i]['attention_mask'].to(self.device)  # (batch_size, seq_len)
            mask_pos = data[i]['mask_pos'].to(self.device)  # (batch_size, 2)
            output = self.bert_encoder(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # 2.1 compute the loss based on neighboring triples
            logits = output.logits[mask_pos[:, 0], mask_pos[:, 1], self.entity_begin_idx: self.entity_end_idx]
            loss = self.loss_fc(logits, labels)
            if loss_from_neighbors is None:
                loss_from_neighbors = loss
            else:
                loss_from_neighbors += loss
            # 2.2 get embeddings of mask_token(batch_size, 768)
            mask_embeds = output.hidden_states[-1][mask_pos[:, 0], mask_pos[:, 1], :]
            embeds_from_neighbors.append(mask_embeds)
        loss = loss_from_neighbors / self.neighbor_num
        embeds = torch.stack(embeds_from_neighbors, dim=0)  # (neighbor_num, batch_size, 768)
        embeds = torch.mean(embeds, dim=0)  # (batch_size, 768)

        return loss, embeds

    def triple_classification(self, batch):
        # 1. prepare data
        text_data = batch['text_data']
        input_ids = text_data['input_ids'].to(self.device)
        token_type_ids = text_data['token_type_ids'].to(self.device)
        attention_mask = text_data['attention_mask'].to(self.device)
        mask_pos = text_data['mask_pos'].to(self.device)
       
        code = batch['code']
        # the labels that used to ranking loss
        score_labels = batch['score_labels'].to(self.device)
        # the soft label that used to update
        train_labels = batch['soft_labels'].to(self.device)

        train_labels = score_labels
        # 2. aggregate with neighbors
        # input_embeds: (batch_size, seq_len, hidden_size)
        inputs_embeds = self.bert_encoder.bert.embeddings.word_embeddings(input_ids)

        # 3. encode with bert
        output = self.bert_encoder(inputs_embeds=inputs_embeds, token_type_ids=token_type_ids, attention_mask=attention_mask,output_hidden_states=True)

        # 3. get logits, calculate loss and rank
        x = mask_pos[:,0]
        y = mask_pos[:1]

        last_output = output.hidden_states[-1]
        logits = last_output[mask_pos[:, 0], mask_pos[:, 1]]
        # logits = self.sigmoid(self.mlp(logits))
        logits2 = self.mlp(logits)
        score_labels = torch.unsqueeze(score_labels,1).to(logits.dtype)
        train_labels = torch.unsqueeze(train_labels,1).to(logits.dtype)
        # get loss for backward
        train_loss = self.bce_loss_withlogits(logits2, train_labels)
        # train_loss = self.bce_loss(logits, train_labels)

        batch_loss = train_loss.mean()
        # get loss for ranking
        
        score_loss = self.bce_loss_withlogits(logits2, score_labels)
        score_loss = score_loss.squeeze()

        loss_info = [(c,l) for c,l, label, logit in zip(code, score_loss.tolist(), score_labels,  logits)]
        
        rank = None
        return {'loss': batch_loss,  'rank': rank, 'logits': logits, 'loss_info': loss_info}

    def link_prediction(self, batch):
        # 1. prepare data
        text_data = batch['text_data']
        input_ids = text_data['input_ids'].to(self.device)
        token_type_ids = text_data['token_type_ids'].to(self.device)
        attention_mask = text_data['attention_mask'].to(self.device)
        mask_pos = text_data['mask_pos'].to(self.device)
        labels = batch['labels'].to(self.device)
        code = batch['code']
        
        confidence = batch['soft_labels'].to(self.device)    
        head_input_ids = batch['text_head_prompts']['input_ids'].to(self.device)  # batch_size * 64
        tail_input_ids = batch['text_tail_prompts']['input_ids'].to(self.device)
        head_input_mask = batch['text_head_prompts']['mask_pos'].to(self.device)
        tail_input_mask = batch['text_tail_prompts']['mask_pos'].to(self.device)
        head_labels = batch['head_labels'].to(self.device)  # batch_size
        tail_labels = batch['tail_labels'].to(self.device)
        head_filters = batch['head_filters'].to(self.device)
        tail_filters = batch['tail_filters'].to(self.device)
       
        # input_embeds: (batch_size, seq_len, hidden_size)
        head_inputs_embeds = self.bert_encoder.bert.embeddings.word_embeddings(head_input_ids)
        # 3. encode with bert
        head_output = self.bert_encoder(inputs_embeds=head_inputs_embeds, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states= True)
        # 3. get logits, calculate loss and rank
        head_logits = head_output.logits[head_input_mask[:, 0], head_input_mask[:, 1], self.entity_begin_idx:self.entity_end_idx]
        head_score = self.score_loss_fc_each(head_logits, head_labels)
        head_loss = self.loss_fc_each(head_logits, head_labels)
        last_output = head_output.hidden_states[-1]
        
        tail_token = last_output[tail_input_mask[:,0],tail_input_mask[:, 1],:].unsqueeze(dim = 1)
        head_repre = torch.cat([last_output[:,2:4,:],tail_token], dim = 1)
        
        

        tail_inputs_embeds = self.bert_encoder.bert.embeddings.word_embeddings(tail_input_ids)
        # 3. encode with bert
        tail_output = self.bert_encoder(inputs_embeds=tail_inputs_embeds, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states = True)
        # 3. get logits, calculate loss and rank
        tail_logits = tail_output.logits[tail_input_mask[:, 0], tail_input_mask[:, 1], self.entity_begin_idx:self.entity_end_idx]
        tail_score = self.score_loss_fc_each(tail_logits, tail_labels)
        tail_loss = self.loss_fc_each(tail_logits, tail_labels)

       
            
        tail_logits_score = torch.softmax(tail_logits, dim = -1)[list(range(len(code))), tail_labels]
        head_logits_score = torch.softmax(head_logits, dim = -1)[list(range(len(code))), head_labels]    
            
        last_output = tail_output.hidden_states[-1]
        tail_token = last_output[tail_input_mask[:,0],tail_input_mask[:, 1],:].unsqueeze(dim = 1)
        tail_repre = torch.cat([last_output[:,2:4,:],tail_token], dim = 1)
        
        
        logits_score = head_logits_score + tail_logits_score
        logits = [head_repre , tail_repre]
        loss = confidence * (head_loss + tail_loss)
        score = head_score + tail_score
        train_loss = loss.mean()
        logits_score = [(c,l) for c,l in zip(code, logits_score.tolist())] 
        loss_info = [(c,l) for c,l in zip(code, score.tolist())] 
        return {'loss': train_loss,  'rank': None, 'logits': logits, 'loss_info': loss_info, 'logits_score': logits_score}


    def configure_optimizers(self, total_steps: int):
        parameters = self.get_parameters()
        opt = AdamW(parameters, eps=1e-6)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=opt,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )
        return {'optimizer': opt, 'scheduler': scheduler}

    def get_parameters(self):
        # freeze all layers except word_embeddings when pre-training
        if self.pretraining:
            for n, p in self.bert_encoder.named_parameters():
                if 'word_embeddings' not in n:
                    p.requires_grad = False

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
        if self.pretraining:  # no_decay_param is empty
            return [
                {'params': decay_param, 'weight_decay': 1e-2, 'lr': self.lr}
            ]
        for n, p in self.mlp.named_parameters():
            mlp_param.append(p)

        else:
            return [
                {'params': decay_param, 'weight_decay': 1e-2, 'lr': self.lr},
                {'params': no_decay_param, 'weight_decay': 0, 'lr': self.lr},
                {'params': mlp_param, 'weight_decay': 0, 'lr': self.lr}
            ]

    def freeze(self):
        for n, p in self.bert_encoder.named_parameters():
            p.requires_grad = False


    def save_model(self, save_dir):
        # save_path = os.path.join(save_dir, 'nbert')
        save_path = save_dir
        self.bert_encoder.save_pretrained(save_path)

    def clip_grad_norm(self):
        # this function is useless, because gard norm is less that 1.0
        # raise ValueError('Do not call clip_gram_norm for N-BERT')
        if self.pretraining:
            # print('[WARNING] We do not apply clip_grad_norm for pretrain')
            return
        norms = get_norms(self.bert_encoder.parameters()).item()
        info = f'grads for N-BERT: {round(norms, 4)}'
        clip_grad_norm_(self.bert_encoder.parameters(), max_norm=3.0)

        return info

    def grad_norm(self):
        norms = get_norms(self.bert_encoder.parameters()).item()
        return round(norms, 4)




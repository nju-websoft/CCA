import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from transformers import BertTokenizer, BertForMaskedLM, get_cosine_schedule_with_warmup
from torch.optim.adamw import AdamW
from sklearn.metrics.pairwise import cosine_similarity
class Bert(nn.Module):
    def __init__(self, args: dict, tokenizer: BertTokenizer, bert: BertForMaskedLM):
        super(Bert, self).__init__()

        # 1. hyper params
        self.device = torch.device(args['device'])
        self.lr = args['lm_lr']
        # self.entity_begin_idx = args['text_entity_begin_idx']
        # self.entity_end_idx = args['text_entity_end_idx']
        # 2. model
        self.tokenizer = tokenizer
        self.bert_encoder = bert
        self.bert_encoder.resize_token_embeddings(len(self.tokenizer))
        
        self.mlp = nn.Sequential(
            nn.Linear(768,768),
            nn.ReLU(),
            nn.Linear(768,1)
        )
        self.bce_loss = nn.BCELoss(reduce = False)
        self.sigmoid = nn.Sigmoid()
        self.loss_fc = nn.CrossEntropyLoss(label_smoothing=args['lm_label_smoothing'])
        self.loss_fc_each = nn.CrossEntropyLoss(label_smoothing=args['lm_label_smoothing'], reduce = False)

    def forward(self, batch_data):
        loss = self.triple_classification(batch_data)
        
          
        return loss

    def validate_step(self, batch, batch_idx):
        text_data = batch['text_data']
        input_ids = text_data['input_ids'].to(self.device)
        token_type_ids = text_data['token_type_ids'].to(self.device)
        attention_mask = text_data['attention_mask'].to(self.device)
        mask_pos = text_data['mask_pos'].to(self.device)    
        
        inputs_embeds = self.bert_encoder.bert.embeddings.word_embeddings(input_ids)
        # 3. encode with bert
        output = self.bert_encoder(inputs_embeds=inputs_embeds, token_type_ids=token_type_ids, attention_mask=attention_mask,output_hidden_states=True)
        last_output = output.hidden_states[-1]
        logits = last_output[mask_pos[:, 0], mask_pos[:, 1]]
        logits = self.sigmoid(self.mlp(logits))
        result = logits
    
        return result


    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        return loss 

    def training_epoch_end(self, outputs):
        return np.round(np.mean([loss.item() for loss in outputs]), 2)

    def triple_classification(self, batch):
        # 1. prepare data
        text_data = batch['text_data']
        input_ids = text_data['input_ids'].to(self.device)
        token_type_ids = text_data['token_type_ids'].to(self.device)
        attention_mask = text_data['attention_mask'].to(self.device)
        mask_pos = text_data['mask_pos'].to(self.device)    
        score_labels = batch['labels'].to(self.device)
        inputs_embeds = self.bert_encoder.bert.embeddings.word_embeddings(input_ids)
        # 3. encode with bert
        output = self.bert_encoder(inputs_embeds=inputs_embeds, token_type_ids=token_type_ids, attention_mask=attention_mask,output_hidden_states=True)
        last_output = output.hidden_states[-1]
        logits = last_output[mask_pos[:, 0], mask_pos[:, 1]]
        logits = self.sigmoid(self.mlp(logits))
        labels = torch.unsqueeze(score_labels,1).to(torch.float)
        # get loss for backward
        train_loss = self.bce_loss(logits, labels)
        batch_loss = train_loss.mean()
        return batch_loss
    

    


    def get_word_emb(self, input_idx):
        m = len(input_idx)
        input_embeds = self.bert_encoder.bert.embeddings.word_embeddings(input_idx)
        input_embeds = input_embeds.detach().numpy() 
        sim1 = torch.tensor(cosine_similarity(input_embeds)) # len * len
        t = 0.1
        sim2 = sim1/t
        # mask = np.ones((m,m),dtype=int) - np.eye(m, dtype=int)
        # sim = sim*mask
        mask = torch.tensor(np.eye(m, dtype=int),dtype=torch.bool)
        sim1 = sim1.masked_fill(mask, -1e9)
        sim2 = sim2.masked_fill(mask, -1e9)
        p_attn1 = nn.functional.softmax(sim1, dim=1)
        p_attn2 = nn.functional.softmax(sim2, dim=1)

        return p_attn2
        # max_similar_pos = np.argmax(sim1, axis = 1)
        # max_similar_idx = [input_idx[i] for i in max_similar_pos]
        #  return max_similar_pos


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
        decay_param = []
        no_decay_param = []
        for n, p in self.bert_encoder.named_parameters():
            if not p.requires_grad:
                continue
            if ('bias' in n) or ('LayerNorm.weight' in n):
                no_decay_param.append(p)
            else:
                decay_param.append(p)
        
        return [
            {'params': decay_param, 'weight_decay': 1e-2, 'lr': self.lr},
            {'params': no_decay_param, 'weight_decay': 0, 'lr': self.lr}
        ]

    def freeze(self):
        for n, p in self.bert_encoder.named_parameters():
            p.requires_grad = False

    # 保存模型
    def save_model(self, save_dir):
        save_path = os.path.join(save_dir, 'nbert')
        self.bert_encoder.save_pretrained(save_path)





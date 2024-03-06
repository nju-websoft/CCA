import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
from torch.cuda.amp import autocast
from .utils import get_ranks, get_norms, get_scores
from .nbert import NBert
from .nformer import NFormer
from torch.optim.adamw import AdamW
import math
import random
class Inter_Classifier(nn.Module):
    def __init__(self, args: dict):
        super(Inter_Classifier, self).__init__()
        self.use_concatenate = args['use_concatenate']
        self.use_contrastive = True
        self.device = torch.device(args['device'])
        self.contrastive_lr = args['contrastive_lr']
        self.concatenate_lr = args['concatenate_lr']

        self.text_projection = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768,  256)
            )
        self.struc_projection = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear( 256, 256)
            )
        
        self.device = torch.device(args['device'])
        self.mlp = nn.Sequential(
            nn.Linear(1536,768),
            nn.ReLU(),
            nn.Linear(768,1)
        )
       

        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCELoss(reduce = False)
        self.bce_loss_withlogits = nn.BCEWithLogitsLoss(reduce = False)
       
        self.T = 0.25
        self.lam = 1

    def forward(self, text_emb, struc_emb, batch, text_logits_score, struc_logits_score):
        if self.use_concatenate and self.use_contrastive:
            prediction_output = self.joint_prediction(text_emb, struc_emb, batch)
            contrastive_output = self.inter_contrastive(text_emb, struc_emb, batch)
            alpha = 1
            beta = 1
            loss = prediction_output['loss'] + alpha * contrastive_output['loss']
            loss_info = prediction_output['loss_info']
            for i in range(len(batch)):
                loss_info[i][1] += beta * contrastive_output['loss_info'][i][1] 
            output = {'loss': loss, 'loss_info': loss_info, 'logits':prediction_output['logits']}
        elif self.use_concatenate:
            output = self.concatenate_prediction(text_emb, struc_emb, batch)
        elif self.use_contrastive:
            output = self.contrastive_prediction(text_emb, struc_emb, batch, text_logits_score, struc_logits_score)

        return output['loss'], output['loss_info'],output['logits']


    
    def contrastive_prediction(self, text_emb, struc_emb, batch, text_logits_score, struc_logits_score):
        code = batch['code']

        
        head_text_emb = self.text_projection(text_emb[0])
        tail_text_emb = self.text_projection(text_emb[1])
        head_struc_emb = self.struc_projection(struc_emb[0])
        tail_struc_emb = self.struc_projection(struc_emb[1])
                
        batch_size, _, emb_size = head_text_emb.shape
        confidence = batch['soft_labels'].to(self.device)
 
        pos_head_text_emb = head_text_emb.reshape(batch_size, 3* emb_size)
        pos_head_struc_emb = head_struc_emb.reshape(batch_size, 3* emb_size)
        pos_tail_text_emb = tail_text_emb.reshape(batch_size, 3* emb_size)
        pos_tail_struc_emb = tail_struc_emb.reshape(batch_size, 3* emb_size)

        interactive_loss = 0
        interactive_score_similarity = 0
        self.T = 0.1
        mask = torch.eye(batch_size).to(self.device)
        for j in range(2):
            if j==0:
                text_emb2 = (head_text_emb )
                struc_emb2 = (tail_struc_emb )             
            else:
                text_emb2 = tail_text_emb
                struc_emb2 = head_struc_emb
                                
            pos_text_emb = text_emb2.reshape(batch_size, 3* emb_size)
            pos_struc_emb = struc_emb2.reshape(batch_size, 3* emb_size)
        
            text_neg_list = []
            struc_neg_list = []
                        
            seq = list(range(batch_size)) 
   
            for k in range(2):          
                random.shuffle(seq) 
                neg_text_emb = torch.cat( [text_emb2[:, [0,1], :], text_emb2[seq, 2, :].unsqueeze(1)], dim = 1).reshape(batch_size, 3* emb_size)
                neg_struc_emb = torch.cat([struc_emb2[:, [0,1], :] , struc_emb2[seq, 2, :].unsqueeze(1)], dim = 1).reshape(batch_size, 3* emb_size)
                text_neg_list.append(neg_text_emb)
                struc_neg_list.append(neg_struc_emb)
                random.shuffle(seq)
                neg_text_emb = torch.cat([text_emb2[seq, 0, :].unsqueeze(1), text_emb2[:, [1,2], :], ], dim = 1).reshape(batch_size, 3* emb_size)
                neg_struc_emb = torch.cat([struc_emb2[seq, 0, :].unsqueeze(1), struc_emb2[:, [1,2], :], ], dim = 1).reshape(batch_size, 3* emb_size)            
                text_neg_list.append(neg_text_emb)
                struc_neg_list.append(neg_struc_emb)

            for k in range(2):
                        
                random.shuffle(seq)
                neg_text_emb = text_emb2[seq, :, :].reshape(batch_size, 3* emb_size)
                random.shuffle(seq)
                neg_struc_emb = struc_emb2[seq, :, :].reshape(batch_size, 3* emb_size)   
                text_neg_list.append(neg_text_emb)
                struc_neg_list.append(pos_struc_emb)
                
                struc_neg_list.append(neg_struc_emb) 
                text_neg_list.append(pos_text_emb)   


            text_neg_tensor = torch.stack(text_neg_list)
            struc_neg_tensor = torch.stack(struc_neg_list)     
            neg_similarity = torch.exp(torch.nn.functional.cosine_similarity(text_neg_tensor, struc_neg_tensor, dim = 2))
            neg_similarity = torch.sum(neg_similarity, dim = 0) # / len(text_neg_list)      
            pos_similarity = torch.exp(  torch.nn.functional.cosine_similarity(pos_text_emb, pos_struc_emb, dim =1))  
    
            interactive_loss += - confidence * torch.log(  pos_similarity/ (pos_similarity + neg_similarity))
            interactive_score_similarity += - pos_similarity
        
        inner_loss = 0
        inner_score_similarity = 0
 
        inner_loss = 0  
        for j in range(2):
            if j==0:
                inner_emb1 = (head_text_emb )
                inner_emb2 = (tail_text_emb )             
            else:
                inner_emb1 = (head_struc_emb )
                inner_emb2 = (tail_struc_emb )  
                                
            inner_pos_emb1 = inner_emb1.reshape(batch_size, 3* emb_size)
            inner_pos_emb2 = inner_emb2.reshape(batch_size, 3* emb_size)
            
            neg_list1 = []
            neg_list2 = []
                        
            seq = list(range(batch_size)) 
            random.shuffle(seq)
   
            for k in range(4):          
                random.shuffle(seq) 
                neg_emb1 = torch.cat( [inner_emb1[:, [0,1], :], inner_emb1[seq, 2, :].unsqueeze(1)], dim = 1).reshape(batch_size, 3* emb_size)
                neg_emb2 = torch.cat([inner_emb2[:, [0,1], :] , inner_emb2[seq, 2, :].unsqueeze(1)], dim = 1).reshape(batch_size, 3* emb_size)
                neg_list1.append(neg_emb1)
                neg_list2.append(neg_emb2)

                random.shuffle(seq)
                neg_emb1 = torch.cat([inner_emb1[seq, 0, :].unsqueeze(1), inner_emb1[:, [1,2], :], ], dim = 1).reshape(batch_size, 3* emb_size)
                neg_emb2 = torch.cat([inner_emb2[seq, 0, :].unsqueeze(1), inner_emb2[:, [1,2], :], ], dim = 1).reshape(batch_size, 3* emb_size)            
                neg_list1.append(neg_emb1)
                neg_list2.append(neg_emb2)
            
            for k in range(4):          

                random.shuffle(seq)
                neg_emb1 = inner_emb1[seq, :, :].reshape(batch_size, 3* emb_size)
                random.shuffle(seq)
                neg_emb2 = inner_emb2[seq, :, :].reshape(batch_size, 3* emb_size)
                
                neg_list1.append(inner_pos_emb2)
                neg_list2.append(neg_emb1)
                
                neg_list1.append(inner_pos_emb1) 
                neg_list2.append(neg_emb2) 

            neg_tensor1 = torch.stack(neg_list1)
            neg_tensor2 = torch.stack(neg_list2)     
            neg_similarity = torch.exp(torch.nn.functional.cosine_similarity(neg_tensor1, neg_tensor2, dim = 2))
            neg_similarity = torch.sum(neg_similarity, dim = 0) # / len(text_neg_list) 
                
            pos_similarity = torch.exp(  torch.nn.functional.cosine_similarity(inner_pos_emb1, inner_pos_emb2, dim =1))  
            inner_loss += - confidence * torch.log(  pos_similarity/ (pos_similarity + neg_similarity))
            inner_score_similarity += - pos_similarity
                
        final_score_similarity = interactive_score_similarity #+ inner_score_similarity
        batch_loss = (interactive_loss).mean()

        loss_info = [(c,l) for c,l in zip(code, final_score_similarity.tolist() )]
        logits = [(c,text, struc) for c, text, struc in zip(code, (pos_head_text_emb + pos_tail_text_emb).to('cpu').detach(), (pos_head_struc_emb + pos_tail_struc_emb).to('cpu').detach())]

 
        return {'loss': batch_loss,   'loss_info': loss_info, 'logits': logits}
   

    def sim_matrix(self, a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.clamp(a_n, min=eps)
        b_norm = b / torch.clamp(b_n, min=eps)
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        sim_mt = torch.exp(sim_mt)
        return sim_mt



    def get_parameters(self):
        contrastive_param = []
        concatenate_param = []
        for n, p in self.named_parameters():
            if 'text_projection' in n or 'struc_projection' in n:
                contrastive_param.append(p)
            else:
                concatenate_param.append(p)
        return [
            {'params': contrastive_param, 'weight_decay': 0, 'lr': self.contrastive_lr},
            {'params': concatenate_param, 'weight_decay': 0, 'lr': self.concatenate_lr}
        ]


    def concatenate_prediction(self, text_emb, struc_emb, batch):
        concatenate_emb = torch.cat([text_emb, struc_emb],1)
        logits = self.mlp(concatenate_emb)

        score_labels = batch['score_labels'].to(self.device)

        train_labels = batch['soft_labels'].to(self.device)
        code = batch['code']

        score_labels = torch.unsqueeze(score_labels,1).to(logits.dtype)
        train_labels = torch.unsqueeze(train_labels,1).to(logits.dtype)

  
        train_loss = self.bce_loss_withlogits(logits, train_labels)
        batch_loss = train_loss.mean()

        score_loss = self.bce_loss_withlogits(logits, score_labels)
        score_loss = score_loss.squeeze()
        loss_info = [(c,l) for c,l in zip(code, score_loss.tolist())] 

        rank = None
        return {'loss': batch_loss,   'loss_info': loss_info, 'logits':concatenate_emb}
   
   
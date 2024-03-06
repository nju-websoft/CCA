import argparse
import json
import os
import random
import shutil
# from Code import KGCDataModule, NBert
from time import localtime, strftime, time
from random import shuffle
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from bert import Bert

from TransE import TransE
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'



class Similarity_Trainer:
    def __init__(self, anomaly_type, triples, config: dict, id2ent, id2rel, ent2id, rel2id, entities, relations, r2h_set = None, r2t_set = None): 

        
        # self.pretraining_path = config['pretraining_path']
        # self.output_path = config['output_path']
        # self.epoch = config['epoch']
        # self.result_path = config['result_path']
        self.triples = triples
        self.anomaly_type = anomaly_type
        self.entities = entities
        self.relations = relations
        self.tokenizer = None
        self.config = config
        self.ori_triples_set = set(triples)
        self.id2rel = id2rel
        self.id2ent = id2ent
        self.ent2id = ent2id
        self.rel2id = rel2id
        self.r2h_set = r2h_set
        self.r2t_set = r2t_set
        self.e2neighbor, self.e2relneighbor = self.get_neighbor()
        self.device = config['device']
        self.max_seq_length = config['max_seq_length']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.pin_memory = config['pin_memory']


        if self.anomaly_type == 3:
            self.anomaly_triples = []
            self.selected_triples = []
            self.epoch= 200
            self.topk = 10
            # read dataset
            # initialize anomaly
            self.neg_ratio = 5
            self.test_ratio = 0.2
            self.model = None
            self.optimizer = None
            # prepare initial data
            
    def get_neighbor(self):
        e2neighbor = {}
        e2relneighbor = {}
        for h,r,t in self.triples:
            if h in e2neighbor.keys():
                e2neighbor[h].append(t)
            else:
                e2neighbor[h] = [t]
            if t in e2neighbor.keys():
                e2neighbor[t].append(h)
            else:
                e2neighbor[t] = [h]
            if h in e2relneighbor.keys():
                e2relneighbor[h].append((r,t))
            else:
                e2relneighbor[h] = [(r,t)]
            if t in e2relneighbor.keys():
                e2relneighbor[t].append((r,h))
            else:
                e2relneighbor[t] = [(r,h)]
        return e2neighbor, e2relneighbor
    

    def adversarial_train(self, num_anomalies):     
        i = 0
        while i <=50:
            i += 1 
            # prepare the dataset
            train_set, test_set = self.construct_dataset(self.triples, self.neg_ratio, self.test_ratio)
            # prepare transE model
            model = TransE(len(self.entities), len(self.relations), self.device).to(self.device)
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0)
            # get test neg result
            # [selected_triples, neg_triples]
            self.transE_train(model, optimizer, train_set, test_set)
            selected, candidate = self.transE_test(model, optimizer, train_set, test_set)
            # update result
            self.result_update(selected, candidate)
            if len(self.anomaly_triples) > 1.4 * num_anomalies:
                break  
        return self.selected_triples, self.anomaly_triples
     
    def result_update(self, selected, candidate):
        anomaly_dict = {}
        for i, (h,r,t) in enumerate(candidate):
            if h not in anomaly_dict.keys():
                anomaly_dict[h] = [(h,r,t)]
                self.anomaly_triples.append((h,r,t))
                self.selected_triples.append(selected[i])
            elif (h,r,t) not in anomaly_dict[h]:
                anomaly_dict[h].append((h,r,t))
                self.anomaly_triples.append((h,r,t))
                self.selected_triples.append(selected[i])


    def get_similar_result(self):
        # initialize the model
        tokenizer_path = self.config['model_path']
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_basic_tokenize=False)
        text_offset = self.resize_tokenizer()
        self.model = self._load_model(self.config, self.tokenizer)

        r2t_set = self.r2t_set
        r2h_set = self.r2h_set
        r2t_similar_index = {}
        r2h_similar_index = {}
        # replace the entity
        for rel in tqdm(r2t_set.keys()):
            # relation to tail
            entities = r2t_set[rel]
            entity_name = [self.entities[ent]['name'] for ent in entities]
            input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(entity_name))
            x = self.model.get_word_emb(input_ids)
            r2t_similar_index[rel] = x
         
        for rel in tqdm(r2h_set.keys()):
            # relation to head
            entities = r2h_set[rel]
            entity_name = [self.entities[ent]['name'] for ent in entities]
            input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(entity_name))
            x = self.model.get_word_emb(input_ids)
            r2h_similar_index[rel] = x
        
        return r2h_similar_index, r2t_similar_index

    def transE_train(self,model, optimizer, train_set, test_set):
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)  
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=True)  
        batch_num = int(len(train_set)/self.batch_size)
        for epoch in range(self.epoch):            
            entity_norm = torch.norm(model.entity_embedding.weight.data, dim=1, keepdim=True)
            model.entity_embedding.weight.data = model.entity_embedding.weight.data / entity_norm
            total_loss = 0
            for batch_idx, data in enumerate(train_loader):
                pos = data[0]
                neg_len = len(data)-1
                neg = data[1:neg_len+1]
                pos_head, pos_relation, pos_tail = pos[0], pos[1], pos[2]
                neg_loss = 0
                for i in range(neg_len):
                    neg_head, neg_relation, neg_tail = neg[i][0], neg[i][1], neg[i][2]                    
                    loss = model(pos_head.to(self.device), pos_relation.to(self.device), pos_tail.to(self.device),
                                  neg_head.to(self.device), neg_relation.to(self.device), neg_tail.to(self.device))
                    neg_loss += loss
                batch_loss = neg_loss/neg_len
                total_loss += batch_loss.item()
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
            print('epoch: '+str(epoch) + ' train loss: ' + str(total_loss/batch_num) + ' ')

    def transE_test(self,model, optimizer, train_set, test_set):
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=True)  
        batch_num = len(train_set)/self.batch_size
        selected_triplets = []
        candidate_anomaly = []
        topk = self.topk
        total_head_hit_num = 0
        total_tail_hit_num = 0
        for batch_idx, data in enumerate(test_loader):

            head, relation, tail = data[0], data[1], data[2]
            head = head.to(self.device)
            relation = relation.to(self.device)
            tail = tail.to(self.device)
            head_hit_num, head_hit_predict, tail_hit_num, tail_hit_predict = model.head_tail_predict(head, relation, tail,topk)
            head_hit_predict = head_hit_predict.cpu().numpy().tolist()
            tail_hit_predict = tail_hit_predict.cpu().numpy().tolist()
            head = head.cpu().numpy().tolist()
            tail = tail.cpu().numpy().tolist()
            relation = relation.cpu().numpy().tolist()
            selected, candidates = self.get_candidate_anomaly(head_hit_predict, tail_hit_predict, 
                                                              head, relation, tail)
            selected_triplets += selected
            candidate_anomaly += candidates
            total_head_hit_num += head_hit_num
            total_tail_hit_num += tail_hit_num
        # print('hits@' + str(topk) + ': ' + str(total_head_hit_num/len(test_set)) + ' ' + str(total_tail_hit_num/len(test_set)) )
        return selected_triplets, candidate_anomaly
            
    def get_candidate_anomaly(self, head_hit_predict, tail_hit_predict, head_list, relation_list, tail_list):
        # get the possible wrong answer in TransE
        selected_triplets = []
        candidate_anomaly = []
        for i, candidate_head_list in enumerate(head_hit_predict):
            head = head_list[i]
            tail = tail_list[i]
            relation = relation_list[i]
            random.shuffle(candidate_head_list)
            for candidate_head in candidate_head_list:
                if candidate_head != head and candidate_head != tail and (relation, candidate_head) not in self.e2relneighbor[tail]:
                    selected_triplets.append((head, relation, tail))
                    candidate_anomaly.append((candidate_head, relation, tail))
                    break

        for i, candidate_tail_list in enumerate(tail_hit_predict):
            head = head_list[i]
            tail = tail_list[i]
            relation = relation_list[i]
            random.shuffle(candidate_tail_list)
            for candidate_tail in candidate_tail_list:
                if candidate_tail != tail and candidate_tail != head and (relation,candidate_tail) not in self.e2relneighbor[head]:
                    selected_triplets.append((head, relation, tail))
                    candidate_anomaly.append((head, relation, candidate_tail))
                    break
        return selected_triplets, candidate_anomaly
        


    def validate(self):
        error_num = 0
        correct_num = 0
        low_quality_neg = []
        for batch_idx, batch_data in enumerate(tqdm(self.test_dl)):
            result = self.model.validate_step(batch_data, batch_idx)
            result = [int(result[i] >= 0.5) for i in range(len(result))]
            labels = list(batch_data['labels'])
            batch_error_num = 0
            batch_correct_num = 0
            for i in range(len(result)):
                if result[i] ==0 and labels[i] ==0:
                    low_quality_neg.append(tuple(batch_data['data_id'][i]))
                    batch_correct_num +=1
                elif result[i] ==0 and labels[i] == 1:
                    batch_error_num +=1
                elif result[i] ==1 and labels[i] == 0:
                    batch_error_num +=1
                elif result[i] == 1 and labels[i] == 1:
                    batch_correct_num +=1
                else:
                    print('false')
            error_num += batch_error_num
            correct_num += batch_correct_num

        return error_num, correct_num, low_quality_neg


           


    def _train_one_epoch(self, epoch):
        self.model.train()
        outputs = list()
        all_sample_loss = []
        for batch_idx, batch_data in enumerate(tqdm(self.train_dl)):
            batch_loss = self.model.training_step(batch_data, batch_idx)
            outputs.append(batch_loss)
            # 2. backward
            self.opt.zero_grad()
            batch_loss.backward()
            self.opt.step()
            if self.scheduler is not None:
                self.scheduler.step()



    def construct_dataset(self, triples, neg_ratio, test_ratio):
        selected_triples = []
        num_triples = len(triples)
        num = int(neg_ratio)
        shuffle(triples)
        train_set = []
        test_set = triples[:int(len(triples)*(test_ratio))]
        pos_set = triples[int(len(triples)*(test_ratio)):]
        # get all candidates
        for triplet in pos_set:
            train_data = []
            h,r,t = triplet
            train_data.append((h,r,t,1))
            for i in range(num):
                neg_triplet = self.randomly_replaced(triplet,single=True)
                h,r,t = neg_triplet
                train_data.append((h,r,t,0))
            train_set.append(train_data)   
        return train_set, test_set

    def anomaly_update(self, low_quality_neg, selected_triples, correct_set, anomaly_set):      
        if low_quality_neg == None:
            return selected_triples, correct_set, anomaly_set

        new_selected = []
        new_anomaly_set = []
        for ele in selected_triples:
            selected, anomaly = ele
            if anomaly in low_quality_neg:
                keep_selected = 1
                if keep_selected:
                    new_anomaly_triples = self.randomly_replaced(selected, single = True)
                    new_selected.append([selected, new_anomaly_triples])
                    new_anomaly_set.append(new_anomaly_triples)
            else:
                new_selected.append([selected, anomaly])
                new_anomaly_set.append(anomaly)

        return new_selected, correct_set, new_anomaly_set
        



    def randomly_replaced(self, correct_triples, single= False):
        anomaly_set = []
        num_entity = len(self.id2ent)
        num_relation = len(self.id2rel)
        anomaly = None

        if single:
            head, rel, tail = correct_triples
            replace_pos = random.randint(0, 2)            
            while anomaly in self.ori_triples_set or anomaly == None:
                new_head, new_relation, new_tail = head, rel, tail
                if replace_pos == 0:
                    new_head = random.randint(0, num_entity - 1)
                elif replace_pos == 1:
                    new_relation = random.randint(0, num_relation - 1)
                else:
                    new_tail = random.randint(0, num_entity - 1)
                anomaly = (new_head, new_relation, new_tail)
            return anomaly
        
        for head, rel, tail in correct_triples:       
            while anomaly in self.ori_triples_set or anomaly == None or anomaly in anomaly_set:
                replace_pos = random.randint(0, 2)     
                new_head, new_relation, new_tail = head, rel, tail
                if replace_pos == 0:
                    new_head = random.randint(0, num_entity - 1)
                elif replace_pos == 1:
                    new_relation = random.randint(0, num_relation - 1)
                else:
                    new_tail = random.randint(0, num_entity - 1)
                anomaly = (new_head, new_relation, new_tail)
            anomaly_set.append(anomaly)
            anomaly = None
        return anomaly_set

    def _load_model(self, config: dict, tokenizer: BertTokenizer):
        text_encoder_path = config['model_path']
        print(f'Loading N-Bert from {text_encoder_path}')
        bert_encoder = BertForMaskedLM.from_pretrained(text_encoder_path)
        model = Bert(config, tokenizer, bert_encoder)
        return model

    # def _load_dataset(self, config: dict, train_set, test_set):
    #     config['tokenizer_path'] = config['model_path']
    #     # 1. load tokenizer
    #     tokenizer_path = config['tokenizer_path']
    #     self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_basic_tokenize=False)
    #     text_offset = self.resize_tokenizer()

    #     # 2. resize tokenizer, load datasets
    #     train_examples = self.create_examples(train_set)
    #     test_examples = self.create_examples(test_set)
    #     train_dl = self.get_dataloader(train_examples)
    #     test_dl = self.get_dataloader(test_examples)

    #     return train_dl, test_dl

    def resize_tokenizer(self):
        entity_begin_idx = len(self.tokenizer)
        entity_names = [self.entities[e]['name'] for e in self.entities]
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': entity_names})
        entity_end_idx = len(self.tokenizer)

        relation_begin_idx = len(self.tokenizer)
        relation_names = [self.relations[r]['sep1'] for r in self.relations]
        relation_names += [self.relations[r]['sep2'] for r in self.relations]
        relation_names += [self.relations[r]['sep3'] for r in self.relations]
        relation_names += [self.relations[r]['sep4'] for r in self.relations]
        relation_names += [self.relations[r]['sep5'] for r in self.relations]
      
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': relation_names})
        relation_end_idx = relation_begin_idx + 5 * len(self.relations) 

        return {
            'text_entity_begin_idx': entity_begin_idx,
            'text_entity_end_idx': entity_end_idx,
            'text_relation_begin_idx': relation_begin_idx,
            'text_relation_end_idx': relation_end_idx,
        }


    def collate_fn(self, batch_data):
        prompts = [data_dit['text_prompt'] for data_dit in batch_data] 
        labels = torch.tensor([data_dit['label'] for data_dit in batch_data])
        data_text = [data_dit['data_text'] for data_dit in batch_data]
        data_id = [data_dit['data_id'] for data_dit in batch_data]
        text_data = self.text_batch_encoding(prompts) 
        return {'text_data': text_data, 'data_text': data_text, 'labels': labels, 'prompts': prompts, 'data_id': data_id}

    def get_dataloader(self, dataset):
        dataloader = DataLoader(dataset, collate_fn=self.collate_fn,
                                batch_size=self.batch_size, num_workers=self.num_workers,
                                pin_memory=self.pin_memory, shuffle=True)
        return dataloader

    def text_batch_encoding(self, inputs):
        encoded_data = self.tokenizer(inputs, padding=True, truncation=True, max_length=self.max_seq_length)
        input_ids = torch.tensor(encoded_data['input_ids'])
        token_type_ids = torch.tensor(encoded_data['token_type_ids'])
        attention_mask = torch.tensor(encoded_data['attention_mask'])
        # 第n行第一个为cls       
        mask_pos = torch.nonzero(torch.eq(input_ids, self.tokenizer.cls_token_id))
        return {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask,
                'mask_pos': mask_pos}

    def create_examples(self, dataset):
        """
        :return: {train: [], dev: [], test: []}
        """   
        examples = dict()
        head_entity_set = set()
        tail_entity_set = set()
        relation_set = set()
        data = []
        for (triple,label) in dataset:
            h, r, t = triple
            h = self.id2ent[h]
            t = self.id2ent[t]
            r = self.id2rel[r]
            triple_example = self.create_one_example(h, r, t, label)        
            data.append(triple_example)                             

        return data

    def create_one_example(self, h, r, t, label):
        mask_token = self.tokenizer.mask_token
        sep_token = self.tokenizer.sep_token
        cls_token = self.tokenizer.cls_token  
        head, rel, tail = self.entities[h], self.relations[r], self.entities[t]
        h_name, h_desc = head['name'], head['desc']
        r_name = rel['name']
        t_name, t_desc = tail['name'], tail['desc']
        sep1, sep2, sep3 = rel['sep1'], rel['sep2'], rel['sep3']
        data_id = (self.ent2id[h], self.rel2id[r], self.ent2id[t])
        text_triple_prompt = ' '.join([sep1, h_name, sep2, r_name, sep3, t_name])
        triple_example = {
            'data_triple': (h, r, t),
            'data_text': (head["raw_name"], r_name, tail['raw_name']),
            'text_prompt': text_triple_prompt,
            'label': label,
            'data_id': data_id
        }

        return  triple_example

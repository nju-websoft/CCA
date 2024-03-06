import numpy as np
import random
import torch
import math
import matplotlib.pyplot as plt
from random import shuffle
import os   
import json
import numpy as np
import torch
from tqdm import tqdm
from similarity_trainer import Similarity_Trainer
from transformers import BertForMaskedLM, BertTokenizer
import argparse

class Anomaly_Generator():
    def __init__(self,config):

        self.data_path = config['dataset_path']
        self.dataset_name = config['dataset']

        self.anomaly_ratio = config['anomaly_ratio']
        self.model_path = config['model_path']
        self.device = config['device']
        self.task = config['task']
        self.config = config
        self.ent2id = dict()
        self.id2ent = dict()
        self.rel2id = dict()
        self.id2rel = dict()
        self.h2t = {}
        self.t2h = {}
        # entity to neighbor
        self.e2neighbor = {}
        # relation_map is to store the triplets of one relation for finding similar entity
        self.r2h_set = {}
        self.r2t_set = {}

        self.e2neighbor_secondary = {}
        # calculate the data
        self.r2num = dict()
        self.e2num = dict()

        self.entities, self.relations = self.read_support()
        print(f'Number of entities: {len(self.entities)}; Number of relations: {len(self.relations)}')

        # 2.1 expand the tokenizer for BERT
       

        self.lines = self.read_lines()
        self.num_entity = len(self.id2ent)
        self.num_relation = len(self.rel2id)
        self.triples = self.lines['train']
        self.num_original_triples = len(self.triples)

        self.triple_ori_set = set(self.triples)
        self.num_anomalies = int((self.anomaly_ratio * self.num_original_triples) / (1 - self.anomaly_ratio))
        # self.num_anomalies = int(self.anomaly_ratio * self.num_original_triples)

        self.selected_triples, self.anomalies = self.inject_anomaly()            
        self.calculate_distribution()
        self.output()
        
                
    
            
    def get_next_hop_neighbor(self, origin_hop_neighbor):
        next_hop_neighbor = {}
        for (h, r, t) in tqdm(self.triples):
            if h in next_hop_neighbor.keys():                
                next_hop_neighbor[h] = next_hop_neighbor[h].union(origin_hop_neighbor[t])
            else:
                next_hop_neighbor[h] = origin_hop_neighbor[t]
                
            if t in next_hop_neighbor.keys():                
                next_hop_neighbor[t] = next_hop_neighbor[t].union(origin_hop_neighbor[h])
            else:
                next_hop_neighbor[t] = origin_hop_neighbor[h]
                
        return next_hop_neighbor    
            

        


            
            
            
        
        
   
    
    def calculate_distribution(self):
        n2rel = dict()
        n2ent = dict()
        for key, value in self.r2num.items():
            if value not in n2rel.keys():
                n2rel[value] = 0
            n2rel[value]+=1
        for key, value in self.e2num.items():
            if value not in n2ent.keys():
                n2ent[value] = 0
            n2ent[value]+=1


        
            
    def get_add_ent_id(self, ent, add=True):
        if ent in self.ent2id:
            ent_id = self.ent2id[ent]
        elif add:
            ent_id = len(self.ent2id)
            self.ent2id[ent] = ent_id
            self.id2ent[ent_id] = ent
        else:
            #print(ent)
            ent_id=-1

        return ent_id

    def get_add_rel_id(self, rel):
        if rel in self.rel2id:
            rel_id = self.rel2id[rel]
        else:
            rel_id = len(self.rel2id)
            self.rel2id[rel] = rel_id
            self.id2rel[rel_id] = rel
        return rel_id

    def output(self):
        ratios = [0.05]
        for ratio in ratios:
            num_anomalies = int((ratio * self.num_original_triples) / (1 - ratio))
            type = "mixture_anomaly"
            folder_path = os.path.join(self.data_path,type,str(int(ratio*100)))
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            data_paths = {
                'selected_triples': os.path.join(self.data_path,type,str(int(ratio*100)), 'selected_triples.txt'),
                'anomaly_triples': os.path.join(self.data_path,type,str(int(ratio*100)), 'anomaly_triples.txt'),
            }
            data = self.selected_triples
            for mode in data_paths:
                data_path = data_paths[mode]
                if mode == 'selected_triples':
                    data = self.selected_triples[: num_anomalies]
                else:
                    data = self.anomalies[: num_anomalies]
                with open(data_path, "w") as f:
                    for triple in data:
                        h, r, t = triple
                        print(triple)
                        head = self.id2ent[h]
                        tail = self.id2ent[t]
                        rel = self.id2rel[r]
                        f.write(head + '\t' + rel + '\t' + tail + '\n')

    def read_support(self):
        """
        read entities and relations from files
        :return: two Python Dict objects
        """
        entity_path = os.path.join(self.data_path, 'support', 'entity.json')
        entities = json.load(open(entity_path, 'r', encoding='utf-8'))
        for idx, e in enumerate(entities):  # 14541
            new_name = f'[E_{idx}]'
            raw_name = entities[e]['name']
            desc = entities[e]['desc']
            entities[e] = {
                'token_id': idx,  # used for filtering
                'name': new_name,  # new token to be added in tokenizer because raw name may consist many tokens
                'desc': desc,  # entity description, which improve the performance significantly
                'raw_name': raw_name,  # meaningless for the model, but can be used to print texts for debugging
            }

        relation_path = os.path.join(self.data_path, 'support', 'relation.json')
        relations = json.load(open(relation_path, 'r', encoding='utf-8'))
        for idx, r in enumerate(relations):  # 237
            sep1, sep2, sep3, sep4, sep5 = f'[R_{idx}_SEP1]', f'[R_{idx}_SEP2]', f'[R_{idx}_SEP3]', f'[R_{idx}_SEP4]',  f'[R_{idx}_SEP5]'
            name = relations[r]['name']
            relations[r] = {
                'sep1': sep1,  # sep1 to sep4 are used as soft prompts
                'sep2': sep2,
                'sep3': sep3,
                'sep4': sep4,
                'sep5': sep5,
                'name': name,  # raw name of relations, we do not need new tokens to replace raw names
            }

        return entities, relations

    def read_lines(self):
        """
        read triplets from  files
        :return: a Python Dict, {train: [], dev: [], test: []}
        """
        data_paths = {
            'train': os.path.join(self.data_path, 'train.txt'),
            'dev': os.path.join(self.data_path, 'dev.txt'),
            'test': os.path.join(self.data_path, 'test.txt')
        }

        lines = dict()
        for mode in data_paths:
            data_path = data_paths[mode]
            raw_data = list()

            # 1. read triplets from files
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    h, r, t = str(line).strip().split('\t')
                    raw_data.append((h, r, t))

            # 2. filter triplets which have no textual information
            data = list()
            for h, r, t in raw_data:
                if (h in self.entities) and (t in self.entities) and (r in self.relations):
                    head_id = self.get_add_ent_id(h)
                    rel_id = self.get_add_rel_id(r)
                    tail_id = self.get_add_ent_id(t)
                    data.append((head_id, rel_id, tail_id))
                    if rel_id not in self.r2h_set.keys():
                        self.r2h_set[rel_id] = set()
                        self.r2t_set[rel_id] = set()
                    self.r2h_set[rel_id].add(h)
                    self.r2t_set[rel_id].add(t)

                    if rel_id not in self.r2num.keys():
                        self.r2num[rel_id] = 0
                    if head_id not in self.e2num.keys():
                        self.e2num[head_id] = 0
                    if tail_id not in self.r2num.keys():
                        self.e2num[tail_id] = 0
                    self.r2num[rel_id] += 1
                    self.e2num[head_id] += 1
                    self.e2num[tail_id] += 1 
                    if head_id in self.e2neighbor.keys():
                        self.e2neighbor[head_id].add(tail_id)
                    else:
                        self.e2neighbor[head_id]= set()
                        self.e2neighbor[head_id].add(tail_id)
                        
                    if tail_id in self.e2neighbor.keys():
                        self.e2neighbor[tail_id].add(head_id)
                    else:
                        self.e2neighbor[tail_id]= set()
                        self.e2neighbor[tail_id].add(head_id)
                    
                    
            if len(raw_data) > len(data):
                raise ValueError('There are some triplets missing textual information')
            lines[mode] = data
        for rel in self.r2h_set.keys():
            self.r2h_set[rel] = list(self.r2h_set[rel])
            self.r2t_set[rel] = list(self.r2t_set[rel])
        lines['train'] = lines['train'] + lines['dev'] + lines['test']

        return lines

    def get_vocab(self):
        """
        construct the vocab for our KGE module
        :return: two Python Dict
        """
        tokens = ['[CLS]','[PAD]', '[MASK]', '[SEP]', self.no_relation_token]
        entity_names = [e for e in self.entities]
        relation_names = []
        for r in self.relations:
            relation_names += [r, f'{r}_reverse']

        entity_begin_idx = len(tokens)
        entity_end_idx = len(tokens) + len(entity_names)
        relation_begin_idx = len(tokens) + len(entity_names)
        relation_end_idx = len(tokens) + len(entity_names) + len(relation_names)

        tokens = tokens + entity_names + relation_names
        vocab = dict()
        for idx, token in enumerate(tokens):
            vocab[token] = idx

        return vocab, {
            'struc_entity_begin_idx': entity_begin_idx,
            'struc_entity_end_idx': entity_end_idx,
            'struc_relation_begin_idx': relation_begin_idx,
            'struc_relation_end_idx': relation_end_idx,
        }

    def delete_invalid_pos_triples(self, pos_triples):

        new_pos_triples = []
        for head, rel, tail in pos_triples:
            if len(self.r2h_set[rel]) <= 5 or len(self.r2t_set[rel]) <= 5:
                continue
            new_pos_triples.append((head, rel, tail))
        return new_pos_triples




    def inject_anomaly(self):
        print("Inject anomalies!")
        original_triples = self.triples
        triple_size = len(original_triples)
        selected_tripls = None
        sample_num = self.num_anomalies//3
        # randomly replaced
        idx = random.sample(range(0, self.num_original_triples - 1), int(sample_num * 1.1))
        selected_triples1 = [original_triples[idx[i]] for i in range(len(idx))]
        # replaced entities randomly
        anomalies1 = self.randomly_replaced(selected_triples1) 
        
        # similarly replaced
        idx = random.sample(range(0, self.num_original_triples - 1), int(sample_num*1.1))
        selected_triples2 = [original_triples[idx[i]] for i in range(len(idx))]
        selected_triples2, anomalies2 = self.similarly_bert_replaced(selected_triples2)
        selected_triples2 = selected_triples2[:int(sample_num * 1.1)]
        anomalies2 = anomalies2[:int(sample_num*1.1)]
        
        # adversarial replaced
        # use TransE to get the similar entity
        selected_triples3, anomalies3 = self.adversarial_replaced(int(sample_num * 1.1))
        selected_triples3 = selected_triples3[:int(sample_num * 1.1)]
        anomalies3 = anomalies3[:int(sample_num*1.1)]
        
        candidate_selected_list = selected_triples1 + selected_triples2 + selected_triples3
        candidate_anomaly_list = anomalies1 + anomalies2 + anomalies3
        final_anomaly_set = set()
        final_anomaly_list = []
        final_selected_list = []

        idxs = random.sample(range(0, len(candidate_anomaly_list)), len(candidate_anomaly_list))

        for i in idxs:
            (h, r, t) = candidate_anomaly_list[i]
            if (h, r, t) not in final_anomaly_set:
                final_anomaly_set.add((h, r, t))
                final_anomaly_list.append((h,r,t))
                final_selected_list.append(candidate_selected_list[i])
        selected_triples = final_selected_list[:self.num_anomalies]
        anomalies = final_anomaly_list[:self.num_anomalies]
        return selected_triples, anomalies

    def adversarial_replaced(self,num_anomalies):
        trainer = Similarity_Trainer(3, self.triples, self.config, self.id2ent, self.id2rel, self.ent2id, self.rel2id, self.entities, self.relations)
        selected_triples, anomaly_triples = trainer.adversarial_train(num_anomalies)
        selected = [(selected_triples[i], anomaly_triples[i]) for i in range(len(selected_triples))]
        random.shuffle(selected)
        selected_triples = [sample[0] for sample in selected]
        anomaly_triples = [sample[1] for sample in selected]
        return selected_triples, anomaly_triples



    def randomly_replaced(self, correct_triples):
        anomaly_set = []
        anomaly = None
        for head, rel, tail in correct_triples:
            replace_pos = random.randint(0, 2)            
            while anomaly in self.triple_ori_set or anomaly == None:
                new_head, new_relation, new_tail = head, rel, tail
                if replace_pos == 0:
                    new_head = random.randint(0, self.num_entity - 1)
                elif replace_pos == 1:
                    new_relation = random.randint(0, self.num_relation - 1)
                else:
                    new_tail = random.randint(0, self.num_entity - 1)
                anomaly = (new_head, new_relation, new_tail)
            anomaly_set.append(anomaly)
            anomaly = None
        return anomaly_set

    def similarly_bert_replaced(self, selected_triples):
        trainer =  Similarity_Trainer(2, self.triples, self.config, self.id2ent, self.id2rel, self.ent2id, self.rel2id, self.entities, self.relations, self.r2h_set, self.r2t_set)
        r2h_similar_index, r2t_similar_index = trainer.get_similar_result()
        selected_triples, neg_triples = self.similarly_replaced(selected_triples, r2h_similar_index, r2t_similar_index)
        return selected_triples, neg_triples
        




    def similarly_replaced(self, correct_triples, r2h_similar_index=None, r2t_similar_index=None):
        neg_triples = []
        new_selected_triples = []
        anomaly = None
        for head, rel, tail in tqdm(correct_triples):
            anomaly = None
            head_list = self.r2h_set[rel]
            tail_list = self.r2t_set[rel]
            head_pos = head_list.index(self.id2ent[head])
            tail_pos = tail_list.index(self.id2ent[tail])
            count = 0
            while anomaly in self.triple_ori_set or anomaly == None:
                count += 1
                new_head, new_relation, new_tail = head, rel, tail
                if len(head_list) == 1:
                    replaced_pos == 1
                elif len(tail_list) == 1:
                    replaced_pos == 0
                else:
                    replaced_pos = random.randint(0, 1)

                if replaced_pos == 0:
                    if r2h_similar_index == None:
                        similar_index = random.randint(0, len(head_list) - 1)
                    else:
                        # similar_index = r2h_similar_index[rel][head_pos]
                        prob = r2h_similar_index[rel][head_pos]
                        similar_index = random.choices(list(range(len(head_list))), weights = prob, k=1)[0]
                        # similar_index = random.choices(list(range(len(head_list))),  k=1)[0]
                    new_head = int(self.get_add_ent_id(head_list[similar_index], False))
                else:
                    if r2t_similar_index == None:
                        similar_index = random.randint(0, len(tail_list) - 1)
                    else:
                        # similar_index = r2t_similar_index[rel][tail_pos]
                        prob = r2t_similar_index[rel][tail_pos]
                        similar_index = random.choices(list(range(len(tail_list))), weights = prob, k=1)[0]
                        # similar_index = random.choices(list(range(len(tail_list))),  k=1)[0]
                    new_tail = int(self.get_add_ent_id(tail_list[similar_index], False))
                
                anomaly = (new_head, new_relation, new_tail)
                
                # skip the neighbor:
                # if new_head in self.e2neighbor[new_tail] or new_tail in self.e2neighbor[new_head]:
                #     anomaly = None
                    
                if count >= 30:
                    count = -1
                    break
                # print(anomaly)
            if count != -1:
                neg_triples.append(anomaly)
                new_selected_triples.append((head, rel, tail))
        return new_selected_triples, neg_triples



def get_args( anomaly_ratio):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='fb15k-237', help='select a dataset: fb15k-237 or wn18rr')
    
    parser.add_argument('--model_path', type=str, default='checkpoints/fb15k-237/bert-pretrained')
    parser.add_argument('--device', type=str, default='cuda:1', help='select a gpu like cuda:0')
    parser.add_argument('--task', type=str, default='generate_anomaly')
    parser.add_argument('--max_seq_length', type=int, default=64, help='max sequence length for inputs to bert')
    parser.add_argument('--lm_lr', type=float, default=5e-5, help='learning rate for language model')
    parser.add_argument('--lm_label_smoothing', type=float, default=0.8, help='label smoothing for language model')
    # about the training network structure
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    # 2. some unimportant parameters, only need to change when your server/pc changes, I do not change these
    parser.add_argument('--num_workers', type=int, default=32, help='num workers for Dataloader')
    parser.add_argument('--pin_memory', type=bool, default=True, help='pin memory')


    args = parser.parse_args()
    args = vars(args)
    # args['anomaly_type'] = anomaly_type
    args['anomaly_ratio'] = anomaly_ratio
    root_path = os.path.dirname(__file__)
    root_path = os.path.dirname(root_path)
    root_path = os.path.dirname(root_path)
    path = os.path.join(root_path, 'dataset', args['dataset'])  
    args['dataset_path'] = path
    return args


if __name__ == '__main__':

    anomaly_ratios = [0.05]

    for anomaly_ratio in anomaly_ratios:
        config = get_args(anomaly_ratio)
        anomaly_gen = Anomaly_Generator(config)
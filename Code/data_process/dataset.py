import os
import copy
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import copy
import time
import math

# nothing special in this class
class KGCDataset(Dataset):
    def __init__(self, data: list):
        super(KGCDataset, self).__init__()

        # self.data = copy.deepcopy(data)
        self.data = data
        self.len = len(self.data)
        self.get_code()

    def get_code(self):
        for i in range(self.len):
            self.data[i]['code'] = i
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data[idx]


# prepare the dataset
class KGCDataModule:
    def __init__(self, args: dict, tokenizer, encode_text=False, encode_struc=False):
        # 0. some variables used in this class
        self.task = args['task']
        self.data_path = args['data_path']
        self.batch_size = args['batch_size']
        self.num_workers = args['num_workers']
        self.pin_memory = args['pin_memory']
        self.max_seq_length = args['max_seq_length'] if encode_text else -1

        self.use_bert = args['use_bert']

        self.add_neighbors = args['add_neighbors']

        self.neighbor_num = args['neighbor_num']
        self.neighbor_token = args['neighbor_token']
        self.no_relation_token = args['no_relation_token']

        self.neighbor_idxs = {}

        self.scheme = args['scheme']
        self.encode_text = encode_text
        self.encode_struc = encode_struc

        self.complex = args['complex']
        self.anomaly_ratio = args['anomaly_ratio']
        self.low_degree = True
        # 1. read entities and relations from files
        self.entities, self.relations = self.read_support()
        print(f'Number of entities: {len(self.entities)}; Number of relations: {len(self.relations)}')

        # 2.1 expand the tokenizer for BERT
        self.tokenizer = tokenizer
        text_offset = self.resize_tokenizer()
        # 2.2 construct the vocab for our KGE module
        self.vocab, struc_offset = self.get_vocab()
        # 2.3 offsets indicate the positions of entities in the vocab, we store them in args and pass to other classes
        args.update(text_offset)
        args.update(struc_offset)
        # 2.4 the following two variables will be used to construct the KGE module
        args['vocab_size'] = len(self.vocab)
        args['num_relations'] = struc_offset['struc_relation_end_idx'] - struc_offset['struc_relation_begin_idx']

        # 3.1 read the dataset
        self.lines = self.read_lines()  # {'train': [(h,r,t),...], 'dev': [], 'test': []}
        ### adding the anomalies


        
        # 3.3 entities to be filtered when predict some triplet
        self.entity_filter = self.get_entity_filter()

        # 5. use the triplets in the dataset to construct the inputs for our BERT and KGE module
        if self.task == 'pretrain':
            # utilize entities to get dataset when task is pretrain
            examples = self.create_pretrain_examples()
            train_set = examples['train']+ examples['dev'] + examples['test']
        else:
            examples, self.head_entity_list, self.relation_list, self.tail_entity_list = self.create_examples()

            train_set = examples['train'] + examples['dev'] + examples['test'] + examples['anomaly'] #+ examples['neg']
        self.init_train_set = train_set
        random.shuffle(train_set)
        self.train_ds = KGCDataset(train_set)
        # self.dev_ds = KGCDataset(examples['dev'])
        # self.test_ds = KGCDataset(examples['test'])
        if self.task != 'pretrain':
            self.label = self.get_label(self.train_ds.data,'test')
            self.neighbors = self.get_neighbors()  # {ent: {text_prompt: [], struc_prompt: []}, ...}
            if len(self.label)/len(self.entities) >= 20:
                self.low_degree = False
        else:
            self.label = None
            self.neighbors = None
        


        


    
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
            sep1, sep2, sep3, sep4, sep5 = f'[R_{idx}_SEP1]', f'[R_{idx}_SEP2]', f'[R_{idx}_SEP3]', f'[R_{idx}_SEP4]', f'[R_{idx}_SEP5]'
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

    def resize_tokenizer(self):
        """
        add the new tokens in self.entities and self.relations into the tokenizer of BERT
        :return: a Python Dict, indicating the positions of entities in logtis
        """
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
        relation_names += [self.neighbor_token, self.no_relation_token]
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': relation_names})
        relation_end_idx = relation_begin_idx + 5 * len(self.relations) + 2

        return {
            'text_entity_begin_idx': entity_begin_idx,
            'text_entity_end_idx': entity_end_idx,
            'text_relation_begin_idx': relation_begin_idx,
            'text_relation_end_idx': relation_end_idx,
        }

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

    def read_lines(self):
        """
        read triplets from  files
        :return: a Python Dict, {train: [], dev: [], test: []}
        """
        data_paths = {
            'train': os.path.join(self.data_path, 'train.txt'),
            'dev': os.path.join(self.data_path, 'dev.txt'),
            'test': os.path.join(self.data_path, 'test.txt'),
            'anomaly': os.path.join(self.data_path, self.complex,str(int(self.anomaly_ratio*100)),'anomaly_triples.txt')
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
                    data.append((h, r, t))
            if len(raw_data) > len(data):
                raise ValueError('There are some triplets missing textual information')

            lines[mode] = data


        return lines

    def get_neighbors(self):
        """
        construct neighbor prompts from training dataset
        :return: {entity_id: {text_prompt: [], struc_prompt: []}, ...}
        """
        sep_token = self.tokenizer.sep_token
        mask_token = self.tokenizer.mask_token

        lines = self.lines['train']+ self.lines['dev'] + self.lines['test'] +  self.lines['anomaly']
        lines2 = self.train_ds.data
        data = {e: {'data':[],'text_prompt': [], 'struc_prompt': [], 'code':[]} for e in self.entities}

        for triples_info in lines2:
            h,r,t  = triples_info['data_triple']
            code = triples_info['code']
        # for h, r, t in lines:
            head, rel, tail = self.entities[h], self.relations[r], self.entities[t]
            h_name, r_name, t_name = head['name'], rel['name'], tail['name']
            # sep1, sep2, sep3, sep4 = rel['sep1'], rel['sep2'], rel['sep3'], rel['sep4']

            # 1. neighbor prompt for predicting head entity
            # head_text_prompt = f'{sep1} {mask_token} {sep2} {r_name} {sep3} {t_name} {sep4}'
            triple_struc_prompt = [self.vocab[sep_token], self.vocab[h], self.vocab[r], self.vocab[t]]
            head_struc_prompt = [self.vocab[sep_token],  self.vocab[r], self.vocab[t]]
            data[h]['struc_prompt'].append(head_struc_prompt)
            data[h]['data'].append((h, r, t))
            data[h]['code'].append(code)

            tail_struc_prompt = [self.vocab[sep_token], self.vocab[r], self.vocab[h]]
            data[t]['struc_prompt'].append(tail_struc_prompt)
            data[t]['data'].append((h, r, t))
            data[t]['code'].append(code)

        length = 0
        for x in data:
            length += len(x)
        length = length/len(data)

        # add a fake neighbor if there is no neighbor for the entity
        # for ent in data:
        #     if len(data[ent]['text_prompt']) == 0:
        #         h_name = self.entities[ent]['name']
        #         text_prompt = ' '.join([h_name, sep_token, self.no_relation_token, sep_token, mask_token])
        #         struc_prompt = [self.vocab[ent], self.vocab[self.no_relation_token], self.vocab[mask_token]]
        #         data[ent]['text_prompt'].append(text_prompt)
        #         data[ent]['struc_prompt'].append(struc_prompt)

        return data

    def get_entity_filter(self):
        """
        for given h, r, collect all t
        :return: a Python Dict, {(h, r): [t1, t2, ...]}
        """
        train_lines = self.lines['train']
        dev_lines = self.lines['dev']
        test_lines = self.lines['test']
        lines = train_lines + dev_lines + test_lines

        entity_filter = defaultdict(set)
        for h, r, t in lines:
            entity_filter[h, r].add(self.entities[t]['token_id'])
            entity_filter[t, r].add(self.entities[h]['token_id'])
        return entity_filter

    def create_examples(self):
        """
        :return: {train: [], dev: [], test: []}
        """   
        examples = dict()
        head_entity_set = set()
        tail_entity_set = set()
        relation_set = set()
        
        for mode in self.lines:
            data = list()
            lines = self.lines[mode]
            for h, r, t in tqdm(lines, desc=f'[{mode}]构建examples'):
                head_entity_set.add(h)
                tail_entity_set.add(t)
                relation_set.add(r)
                # todo revise the data.
                triple_example = self.create_one_example(h, r, t)
                if mode == 'anomaly':
                    triple_example['test_label'] = 0
                    triple_example['train_label'] = 1                    
                else:
                    triple_example['test_label'] = 1
                    triple_example['train_label'] = 1         
                data.append(triple_example)                             
            examples[mode] = data
        head_entity_list = list(head_entity_set)
        relation_list = list(relation_set)
        tail_entity_list = list(tail_entity_set)


        return examples, head_entity_list, relation_list, tail_entity_list

    def create_one_example(self, h, r, t):
        mask_token = self.tokenizer.mask_token
        sep_token = self.tokenizer.sep_token
        cls_token = self.tokenizer.cls_token
        neighbor_token = self.neighbor_token
        
        head, rel, tail = self.entities[h], self.relations[r], self.entities[t]
        h_name, h_desc = head['name'], head['desc']
        r_name = rel['name']
        t_name, t_desc = tail['name'], tail['desc']
        sep1, sep2, sep3, sep4, sep5= rel['sep1'], rel['sep2'], rel['sep3'], rel['sep4'], rel['sep5']
        
        # to cut down the description
        h_desc_list = h_desc.split()
        t_desc_list = t_desc.split()
        # desc_bound = int((self.max_seq_length - 10)/2)
        # if len(h_desc_list) + len(t_desc_list)>= (self.max_seq_length - 10) and len(h_desc_list)>=desc_bound:
        #     h_desc = ' '.join(h_desc_list[:desc_bound])
        
        h_desc = ' '.join(h_desc_list[:min(int((self.max_seq_length - 7)),len(h_desc_list))])
        t_desc = ' '.join(t_desc_list[:min(int((self.max_seq_length - 7)),len(t_desc_list))])
        text_triple_prompt = None
        # 1. prepare inputs for nbert
        if self.encode_text:
            text_triple_prompt = ' '.join(
                [sep1, h_name,  sep2, r_name, sep3, t_name,  sep4, h_desc, sep5, t_desc] 
            )
            # kgc
            text_head_prompt = ' '.join(
                [sep1, mask_token,  sep2 , r_name, sep3, t_name, sep4, t_desc]
            )
            
            text_tail_prompt = ' '.join(
                [sep1, h_name,  sep2 , r_name, sep3, mask_token, sep4, h_desc]
            )
          
        else:
            text_triple_prompt = None
            text_head_prompt = None
            text_tail_prompt = None

        if self.encode_struc:
            struc_prompt = [self.vocab[cls_token], self.vocab[h], self.vocab[r], self.vocab[t]]
            struc_tail_prompt = [self.vocab[cls_token], self.vocab[h], self.vocab[r], self.vocab[mask_token]]
            struc_head_prompt = [self.vocab[cls_token], self.vocab[mask_token], self.vocab[r], self.vocab[t]]
        else:
            struc_prompt, struc_head_prompt, struc_tail_prompt = None, None, None

        head_filters = list(self.entity_filter[t, r] - {head['token_id']})
        tail_filters = list(self.entity_filter[h, r] - {tail['token_id']}) 

        triple_example = {
            'data_triple': (h, r, t),
            'data_text': (head["raw_name"], r_name, tail['raw_name']),
            'text_prompt': text_triple_prompt,
            'struc_prompt': struc_prompt,
            'neighbors_label': [head['token_id'], tail['token_id']],

            # struct kgc task
            'struc_head_prompt': struc_head_prompt,
            'struc_tail_prompt': struc_tail_prompt,
            'head_label': head['token_id'],
            'tail_label': tail['token_id'],
            'head_filters': head_filters,
            'tail_filters': tail_filters,

            # text kgc task
            'text_head_prompt': text_head_prompt,
            'text_tail_prompt': text_tail_prompt,

        }

        return  triple_example

    def generate_neg(self,h, r, t):
        neg_triple = (h , r, t)
        head_or_tail = random.randint(0, 2)
        while neg_triple in self.neighbors[h]['data']:
            if head_or_tail == 0:
                neg_triple = (random.choice(self.head_entity_list), r, t)
            elif head_or_tail == 1:
                neg_triple = (h, random.choice(self.relation_list), t)
            else:
                neg_triple = (h, r, random.choice(self.tail_entity_list))
        return neg_triple

    def get_label(self, examples, mode):
        if mode == 'train':
            label = [[example['code'], example['train_label']] for example in examples ]
        else:
            label = [[example['code'], [example['test_label'],example['data_triple']]] for example in examples ]
        return label


    def create_pretrain_examples(self):
        examples = dict()
        for mode in ['train', 'dev', 'test']:
            data = list()
            for h in self.entities.keys():
                name = str(self.entities[h]['name'])
                desc = str(self.entities[h]['desc'])
                desc_tokens = desc.split()

                prompts = [f'The description of {self.tokenizer.mask_token} is that {desc}']
                for i in range(10):
                    begin = random.randint(0, len(desc_tokens))
                    end = min(begin + self.max_seq_length, len(desc_tokens))
                    new_desc = ' '.join(desc_tokens[begin: end])
                    prompts.append(f'The description of {self.tokenizer.mask_token} is that {new_desc}')
                for prompt in prompts:
                    data.append({'prompt': prompt, 'label': self.entities[h]['token_id']})
            examples[mode] = data
        return examples

    def select_neighbor(self, triple_code,  data_list, neighbor_num): 
        neighbor_list = []
        code_list = []
        candidate_neighbor_num = len(data_list['struc_prompt'])
        idxs= []

        for i in range(candidate_neighbor_num):
            if data_list['code'][i] != triple_code:
                idxs.append(i)   
        random.shuffle(idxs)     
        for i in idxs:
            neighbor_list += data_list['struc_prompt'][i % candidate_neighbor_num]
            code_list += [-1, data_list['code'][i % candidate_neighbor_num], data_list['code'][i % candidate_neighbor_num]]
        if len(neighbor_list) < 3 * neighbor_num:
            pad_num = (neighbor_num - int(len(neighbor_list)/3))
            neighbor_list += [1,1,1] * pad_num
            code_list += [-1,-1,-1] * pad_num
        # neighbor_list = sum(neighbor_list, [])
        return neighbor_list[:3*neighbor_num], code_list[:3*neighbor_num]  



       
    
    def text_batch_encoding(self, inputs):
        encoded_data = self.tokenizer(inputs, padding=True, truncation=True, max_length=self.max_seq_length)
        input_ids = torch.tensor(encoded_data['input_ids'])
        test = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        token_type_ids = torch.tensor(encoded_data['token_type_ids'])
        attention_mask = torch.tensor(encoded_data['attention_mask'])
        # [line, column]
        mask_pos = torch.nonzero(torch.eq(input_ids, self.tokenizer.mask_token_id))
       
        return {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask,
                'mask_pos': mask_pos}

    def struc_batch_encoding(self, inputs):
        input_ids = torch.tensor(inputs)
        return {'input_ids': input_ids}

    def collate_fn(self, batch_data):
        if self.task == 'pretrain':
            return self.collate_fn_for_pretrain(batch_data)
        start = time.time()

        # metadata
        data_triple = [data_dit['data_triple'] for data_dit in batch_data]  # [(h, r, t), ...]
        data_text = [data_dit['data_text'] for data_dit in batch_data]  # [(text, text, text), ...]
        data_code = [data_dit['code'] for data_dit in batch_data]
        text_prompts = [data_dit['text_prompt'] for data_dit in batch_data]  # [string, ...]        
        struc_prompts = [data_dit['struc_prompt'] for data_dit in batch_data]  # [string, ...]
        
        labels = [data_dit['train_label'] for data_dit in batch_data]
        soft_labels = [data_dit['train_label'] for data_dit in batch_data]
        
        
        add_neg = False 
        end1 = time.time()
        if add_neg:
            neg_source = copy.deepcopy(data_triple)       
            for h, r, t in neg_source:
                n_h,n_r,n_t = self.generate_neg(h, r, t)
                triple_example = self.create_one_example(n_h, n_r, n_t)
                triple_example['test_label'] = 0
                triple_example['train_label'] = 0            
                data_triple.append(triple_example['data_triple'])
                data_text.append(triple_example['data_text'])
                data_code.append(-1)
                text_prompts.append(triple_example['text_prompt'])
                struc_prompts.append(triple_example['struc_prompt'])
                labels.append(triple_example['train_label'])
                soft_labels.append(triple_example['train_label'])
        
        end2 = time.time()
        labels = torch.tensor(labels)
        soft_labels = torch.tensor(soft_labels)
        struc_data = self.struc_batch_encoding(struc_prompts) if self.encode_struc else None

        if self.encode_text:
            text_head_prompts =  self.text_batch_encoding([data_dit['text_head_prompt'] for data_dit in batch_data])
            text_tail_prompts = self.text_batch_encoding([data_dit['text_tail_prompt'] for data_dit in batch_data])    
            text_data = self.text_batch_encoding(text_prompts)
        else:
            text_head_prompts = None
            text_tail_prompts = None
            text_data = None
        head_struc_neighbors = []
        head_struc_neighbors_code = []
        tail_struc_neighbors = []
        tail_struc_neighbors_code = []
        end3 = time.time()
        if self.add_neighbors:
            batch_struc_neighbors = []        
            for idx, triple in enumerate(data_triple):
                head, relation, tail = triple
                # mask head and use tail entity neighbors
                head_prompt_list, head_code_list = self.select_neighbor(data_code[idx], self.neighbors[tail], self.neighbor_num)  
                # mask tail and use head entity neighbors      
                tail_prompt_list, tail_code_list = self.select_neighbor(data_code[idx], self.neighbors[head], self.neighbor_num)
                head_neighbors_code = [-1, -1 , data_code[idx], data_code[idx]] + head_code_list
                tail_neighbors_code = [-1, data_code[idx], data_code[idx], -1] + tail_code_list
                head_struc_neighbors.append(head_prompt_list)
                head_struc_neighbors_code.append(head_neighbors_code)
                tail_struc_neighbors.append(tail_prompt_list)
                tail_struc_neighbors_code.append(tail_neighbors_code)

            head_struc_neighbors = self.struc_batch_encoding(head_struc_neighbors)
            tail_struc_neighbors = self.struc_batch_encoding(tail_struc_neighbors)
        else:
            head_struc_neighbors = None
            head_struc_neighbors_code = None
            tail_struc_neighbors = None
            tail_struc_neighbors_code = None
        end4 = time.time()
        # kgc task
        head_filters = torch.tensor([[i, j] for i, data_dit in enumerate(batch_data) for j in data_dit['head_filters']])
        tail_filters = torch.tensor([[i, j] for i, data_dit in enumerate(batch_data) for j in data_dit['tail_filters']])
        
        struc_head_prompts =  self.struc_batch_encoding([data_dit['struc_head_prompt'] for data_dit in batch_data])
        struc_tail_prompts = self.struc_batch_encoding([data_dit['struc_tail_prompt'] for data_dit in batch_data])
        head_labels = torch.tensor([data_dit['head_label'] for data_dit in batch_data])
        tail_labels = torch.tensor([data_dit['tail_label'] for data_dit in batch_data])




    
        end = time.time()

        

        return {
            'data': data_triple, 'data_text': data_text,
            'text_data': text_data, 
            'struc_data': struc_data, #'struc_neighbors': batch_struc_neighbors, 'struc_neighbors_code': batch_struc_neighbors_code,
            'neighbor_trustworthy': None, 
            'score_labels': labels,   'code': data_code,
            'soft_labels': soft_labels, 'labels': labels,            
            # kgc task
            'head_filters': head_filters, 'tail_filters':tail_filters, 'struc_head_prompts':struc_head_prompts,
            'struc_tail_prompts':struc_tail_prompts, 'head_labels':head_labels, 'tail_labels': tail_labels,
            'text_tail_prompts': text_tail_prompts, 'text_head_prompts': text_head_prompts,

            # neighbors prompt:
            'head_struc_neighbors': head_struc_neighbors, 'head_struc_neighbors_code': head_struc_neighbors_code,
            'tail_struc_neighbors': tail_struc_neighbors, 'tail_struc_neighbors_code': tail_struc_neighbors_code,
            'head_neighbor_trustworthy': None, 
            'tail_neighbor_trustworthy': None, 
        }

    def collate_fn_for_pretrain(self, batch_data):
        assert self.task == 'pretrain'

        lm_prompts = [data_dit['prompt'] for data_dit in batch_data]  # [string, ...]
        lm_data = self.text_batch_encoding(lm_prompts)

        labels = torch.tensor([data_dit['label'] for data_dit in batch_data])

        return {'text_data': lm_data, 'labels': labels, 'filters': None}

    def get_train_dataloader(self):
        dataloader = DataLoader(self.train_ds, collate_fn=self.collate_fn,
                                batch_size=self.batch_size, num_workers=self.num_workers,
                                pin_memory=self.pin_memory, shuffle=True)
        return dataloader

    def get_dev_dataloader(self):
        dataloader = DataLoader(self.dev_ds, collate_fn=self.collate_fn,
                                batch_size=2 * self.batch_size, num_workers=self.num_workers,
                                pin_memory=self.pin_memory, shuffle=False)
        return dataloader

    def get_test_dataloader(self):
        dataloader = DataLoader(self.test_ds, collate_fn=self.collate_fn,
                                batch_size=2 * self.batch_size, num_workers=self.num_workers,
                                pin_memory=self.pin_memory, shuffle=False)
        return dataloader

    def get_tokenizer(self):
        return self.tokenizer


if __name__ == '__main__':
    pass

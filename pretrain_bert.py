import argparse
import json
import os
import random
import shutil
from Code import KGCDataModule, NBert
from time import localtime, strftime, time

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizer

from utils import score2str

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'



def get_args(complex, anomaly_ratio):
    parser = argparse.ArgumentParser()
    # 1. about training
    parser.add_argument('--task', type=str, default='train', help='pretrain | train | validate')
    parser.add_argument('--model_path', type=str, default='checkpoints/fb15k-237/bert-pretrained')
    parser.add_argument('--epoch', type=int, default=20, help='epoch')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--device', type=str, default='cuda:3', help='select a gpu like cuda:0')
    parser.add_argument('--dataset', type=str, default='fb15k-237', help='select a dataset: fb15k-237 or wn18rr')
    parser.add_argument('--max_seq_length', type=int, default=64, help='max sequence length for inputs to bert')
    # about neighbors
    parser.add_argument('--add_neighbors', action='store_true', default=False)
    parser.add_argument('--neighbor_num', type=int, default=3)
    parser.add_argument('--neighbor_token', type=str, default='[Neighbor]')
    parser.add_argument('--no_relation_token', type=str, default='[R_None]')
    # about text encoder
    parser.add_argument('--lm_lr', type=float, default=5e-5, help='learning rate for language model')
    parser.add_argument('--lm_label_smoothing', type=float, default=0.8, help='label smoothing for language model')
    # about the training network structure
    parser.add_argument('--scheme', type=str, default='mlp', help='select a structure of training')
    # 2. some unimportant parameters, only need to change when your server/pc changes, I do not change these
    parser.add_argument('--num_workers', type=int, default=32, help='num workers for Dataloader')
    parser.add_argument('--pin_memory', type=bool, default=True, help='pin memory')
    parser.add_argument('--use_bert', type=bool, default=True, help='pin memory')


    # 3. convert to dict
    args = parser.parse_args()
    args = vars(args)
    args['add_neighbors'] = False
    # add some paths: tokenzier_path model_path data_path output_path
    root_path = os.path.dirname(__file__)
    args['root_path'] = root_path
    args['pretraining_path'] = os.path.join(root_path, 'checkpoints', args['dataset'],'bert-pretrained')
    # 1. tokenizer path
    
    # 2. model path
    args['model_path'] = os.path.join(root_path, args['model_path'])
    args['tokenizer_path'] = os.path.join(root_path, 'checkpoints', 'bert-base-uncased')
    # 3. data path
    args['data_path'] = os.path.join(root_path, 'dataset', args['dataset']) 

    # 4. output path
    timestamp = strftime('%Y%m%d_%H%M%S', localtime())
    output_dir = os.path.join(root_path, 'output', args['dataset'], 'N-BERT', timestamp)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    args['complex'] = complex
    args['anomaly_ratio'] = anomaly_ratio
    result_dir = os.path.join(root_path, 'result', args['dataset'], 'bert', args['complex'], str(int(args['anomaly_ratio']*100)))
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir)
    args['output_path'] = output_dir
    args['result_path'] = result_dir

    # save hyper params
    with open(os.path.join(args['output_path'], 'args.txt'), 'w') as f:
        json.dump(args, f, indent=4, ensure_ascii=False)

    # set random seed
    seed = 2022
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    return args


class NBertTrainer:
    def __init__(self, config: dict):
        self.is_validate = True if config['task'] == 'validate' else False
        self.pretraining = True if config['task'] == 'pretrain' else False
        self.pretraining_path = config['pretraining_path']
        self.output_path = config['output_path']
        self.epoch = config['epoch']
        self.result_path = config['result_path']
        config['low_degree'] = True
        tokenizer, self.train_dl, self.label = self._load_dataset(config)
        self.model = self._load_model(config, tokenizer).to(config['device'])
        optimizers = self.model.configure_optimizers(total_steps=len(self.train_dl)*self.epoch)
        self.opt, self.scheduler = optimizers['optimizer'], optimizers['scheduler']

        self.soft_label = None
        self.log_path = os.path.join(self.output_path, 'log.txt')
        with open(self.log_path, 'w') as f:
            pass

    def _load_dataset(self, config: dict):
        # 1. load tokenizer
        tokenizer_path = config['tokenizer_path']
        print(f'Loading Tokenizer from {tokenizer_path}')
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_basic_tokenize=False)

        # 2. resize tokenizer, load datasets
        data_module = KGCDataModule(config, tokenizer, encode_text=True)
        tokenizer = data_module.get_tokenizer()
        train_dl = data_module.get_train_dataloader()
        # dev_dl = data_module.get_dev_dataloader()
        # test_dl = data_module.get_test_dataloader()
        label = data_module.label

        return tokenizer, train_dl, label

    def _load_model(self, config: dict, tokenizer: BertTokenizer):
        text_encoder_path = config['model_path']
        print(f'Loading N-Bert from {text_encoder_path}')
        bert_encoder = BertForMaskedLM.from_pretrained(text_encoder_path)
        model = NBert(config, tokenizer, bert_encoder)
        return model

    def _train_one_epoch(self, epoch):
        self.model.train()
        outputs = list()
        all_sample_loss = []
        for batch_idx, batch_data in enumerate(tqdm(self.train_dl)):
            
            batch_loss, sample_loss, _, rank = self.model.training_step(batch_data, batch_idx)
            outputs.append((batch_loss.item(),rank))
            # 2. backward
            self.opt.zero_grad()
            batch_loss.backward()
            self.opt.step()
            if self.scheduler is not None:
                self.scheduler.step()
            if not self.pretraining:
                all_sample_loss += sample_loss
        loss, scores = self.model.training_epoch_end(outputs) 
        return loss, scores


    def train(self):
        best_score = None
        for i in range(1, self.epoch + 1):
            begin_time = time()
            train_loss, scores = self._train_one_epoch(i)
                         # todo ignore the dev and test dataset
            log = f'epoch: {i}, '  + score2str(scores) + '\n'
            print(log)
            if best_score == None or best_score < scores['MRR']:
                best_score = scores['MRR']
                self.model.save_model(self.pretraining_path)

    def validate(self, rank, epoch):
        truth = dict(self.label)
        correct_len = sum([x[0] for x in truth.values()])
        anomaly_len = len(rank) - correct_len
        print('len_anomaly:'+str(anomaly_len))
        topK = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, \
        0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.20, 0.30, 0.4, 0.5, 0.6,0.7,0.8]
        numK = list(map(int,topK*correct_len))
        result_path = os.path.join(self.result_path, str(epoch)+'.txt')
        predict_path = os.path.join(self.result_path, str(epoch)+'_predict'+'.txt')
        with open(result_path,'w') as f:
            for top in topK:
                tp = 0
                fp = 0
                num_k = int (correct_len * top)
                for i in range(num_k):
                    code = rank[i][0]
                    if truth[code][0] == 0:
                        tp = tp + 1
                    else:
                        fp = fp + 1

                recall = tp * 1.0 / anomaly_len
                precision = tp * 1.0 / num_k
                print('epoch: %d, Top%f: precision: %f, recall %f:' %(epoch, top, precision, recall))
                f.write('epoch: %d, Top%f: precision: %f, recall %f:\n' %(epoch, top, precision, recall))
        
        signal = 0 
        with open(predict_path,'w') as f:
            for i in range(correct_len):
                code = rank[i][0]
                if i == numK[signal]:
                    f.write('#' + '\t' + 'top' + '\t' + str(topK[signal]) + '\n')
                    signal = (signal+1) % len(numK)
                if truth[code][0] == 0:
                    triple = truth[code][1]
                    f.write(triple[0] + '\t' + triple[1] + '\t' + triple[2] + '\n')
        return rank
        



    def main(self):
       
        self.train()


if __name__ == '__main__':
    
    
    complex_list = ['mixture_anomaly']
    anomaly_ratios= [0.05]
    for complex in complex_list:
        for anomaly_ratio in anomaly_ratios:
            config = get_args(complex, anomaly_ratio)

            trainer = NBertTrainer(config)
            trainer.main()


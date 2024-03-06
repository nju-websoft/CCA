# Knowledge Graph Error Detection with Contrastive Confidence Adaption

> Knowledge graphs (KGs) often contain various errors. Previous works on detecting errors in KGs mainly rely on triplet embedding from graph structure. We conduct an empirical study and find that these works struggle to discriminate noise from semantically-similar correct triplets. In this paper, we propose a KG error detection model CCA to integrate both textual and graph structural information from triplet reconstruction for better distinguishing semantics. We design interactive contrastive learning to capture the differences between textual and structural patterns. Furthermore, we construct realistic datasets with semantically-similar noise and adversarial noise. Experimental results demonstrate that CCA outperforms state-of-the-art baselines, especially on semanticallysimilar noise and adversarial noise.

## Dependencies

- pytorch==1.10.2
- transformers==4.11.3
- contiguous-params==1.0.0

## Running

For a given dataset, we first construct the noise and then fine-tune a [BERT](https://huggingface.co/bert-base-cased/tree/main) model based on the descriptions of all entities, which will be loaded as a basic module before training CCA. Then we train CCA model to get the result.

### noise construction
```bash
python ./Code/data_process/gen_anomaly.py --model_path checkpoints/bert-base-cased --dataset wn18rr --device cuda:0 --max_seq_length 64 --batch_size 256 --lm_lr 1e-4 --lm_label_smoothing 0.8 --num_workers 8 --pin_memory True 
```



### Pretrain bert encoder with entity description
```bash
python pretrain_bert.py --task pretrain --model_path checkpoints/bert-base-cased --epoch 20 --batch_size 512 --device cuda:0 --dataset wn18rr --max_seq_length 64 --lm_lr 1e-4 --lm_label_smoothing 0.8 --num_workers 8 --pin_memory True 
```


### Train CCA
```bash
python train_CCA.py --epoch 20 --model_path checkpoints/wn18rr/bert-pretrained/ --dataset wn18rr --batch_size 512 --num_workers 32 --use_amp --device cuda:0
```


## Citation
```
@inproceedings{CCA,
  title={Knowledge Graph Error Detection with Contrastive Confidence Adaption},
  author={Liu, Xiangyu and Liu, Yang and Hu, Wei},
  booktitle={AAAI},
  year={2024}
}
```

          
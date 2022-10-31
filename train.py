

import argparse
import random
import os
import numpy as np
import pandas as pd

import torch

from datamodule import DataModule

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from transformers import Trainer, TrainingArguments, HfArgumentParser

from sklearn.metrics import accuracy_score


TOTAL_ASPECT= ['본품#품질', '제품 전체#일반', '본품#일반', '제품 전체#가격', '제품 전체#편의성', '제품 전체#디자인',
       '패키지/구성품#일반', '패키지/구성품#편의성', '제품 전체#품질', '브랜드#가격', '본품#편의성',
       '본품#디자인', '브랜드#일반', '브랜드#품질', '제품 전체#인지도', '패키지/구성품#디자인', '본품#다양성',
       '패키지/구성품#품질', '패키지/구성품#다양성', '브랜드#인지도', '본품#가격', '본품#인지도']

# 본품#품질', '제품 전체#일반' 제외
STAGE2_ASPECT= ['본품#일반', '제품 전체#가격', '제품 전체#편의성', '제품 전체#디자인',
       '패키지/구성품#일반', '패키지/구성품#편의성', '제품 전체#품질', '브랜드#가격', '본품#편의성',
       '본품#디자인', '브랜드#일반', '브랜드#품질', '제품 전체#인지도', '패키지/구성품#디자인', '본품#다양성',
       '패키지/구성품#품질', '패키지/구성품#다양성', '브랜드#인지도', '본품#가격', '본품#인지도', '패키지/구성품#가격']
# 패키지/구성품#가격 : train에는 없는데, dev에는 있음


SENTIMENT= ['negative', 'positive', 'neutral']


stage1_aspect_dict= {'본품#품질': 0, '제품 전체#일반': 1, 'OTHERS': 2}

stage2_num_to_aspect_dict= {idx:value for idx, value in enumerate(STAGE2_ASPECT)}
stage2_aspect_to_num_dict= {value:idx for idx, value in enumerate(STAGE2_ASPECT)}

stage1_sentiment_dict= {'positive': 0, 'OTHERS': 1}
stage2_sentiment_dict= {'negative': 0, 'neutral': 1}




def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_config():
    parser = argparse.ArgumentParser()

    """path, model option"""
    parser.add_argument('--seed', type=int, default=42)
    
    parser.add_argument('--model_name_or_path', type=str, default='klue/roberta-base')
    parser.add_argument('--train_data_path', type=str, default = './data/process_train.csv')
    parser.add_argument('--dev_data_path', type=str, default = './data/process_dev.csv')

    parser.add_argument('--label_columns', type=str, default= 'label_aspect_2')
    parser.add_argument('--max_seq_length', type=int, default= 512)

    parser.add_argument('--config_path', type=str, default = './config.json')
    
    args= parser.parse_args()

    return args

def get_num_classes(args):
  if args.label_columns== 'label_aspect_1':
    args.num_classes= 3
  elif args.label_columns== 'label_aspect_2':
    args.num_classes= len(STAGE2_ASPECT)
    print(len(STAGE2_ASPECT))
  elif args.label_columns== 'label_sentiment_1':
    args.num_classes= 2
  elif args.label_columns== 'label_sentiment_2':
    args.num_classes= 2

  return args

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions
    print('===========================')
    print(preds)
    print('===========================')
    print(labels)
    acc = accuracy_score(labels, preds) 

    return {
        'accuracy': acc,
    }

if __name__ == '__main__':

  args= get_config()
  args.device= 'cuda' if torch.cuda.is_available() else 'cpu'
  seed_everything(args.seed)
  args= get_num_classes(args)

  training_args= HfArgumentParser(TrainingArguments).parse_json_file(args.config_path, allow_extra_keys= True)[0]
  print(training_args)

  tokenizer= AutoTokenizer.from_pretrained(args.model_name_or_path)
  datamodule= DataModule(tokenizer, args)

  print(datamodule.preprocess_train)
  print(datamodule.preprocess_dev)
  config= AutoConfig.from_pretrained(args.model_name_or_path)
  config.num_labels= args.num_classes

  model= AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config= config)
  model.to(args.device)
  trainer= Trainer(
    model= model,
    args= training_args,
    train_dataset= datamodule.preprocess_train,
    eval_dataset= datamodule.preprocess_dev,
    compute_metrics= compute_metrics
  )

  trainer.train()






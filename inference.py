from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import argparse
import random
import os
import numpy as np
import pandas as pd

import torch
import json

from datamodule import DataModule
from collections import defaultdict
from tqdm import tqdm


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


stage1_aspect_to_num_dict= {'본품#품질': 0, '제품 전체#일반': 1, 'OTHERS': 2}
stage1_num_to_aspect_dict= {value:idx for idx, value in stage1_aspect_to_num_dict.items()}

stage2_num_to_aspect_dict= {idx:value for idx, value in enumerate(STAGE2_ASPECT)}
stage2_aspect_to_num_dict= {value:idx for idx, value in enumerate(STAGE2_ASPECT)}

stage1_sentiment_to_num_dict= {'positive': 0, 'OTHERS': 1}
stage1_num_to_sentiment_dict= {value:idx for idx, value in stage1_sentiment_to_num_dict.items()}

stage2_sentiment_to_num_dict= {'negative': 0, 'neutral': 1}
stage2_num_to_sentiment_dict= {value:idx for idx, value in stage2_sentiment_to_num_dict.items()}


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
    parser.add_argument('--test_data_path', type=str, default = './data/nikluge-sa-2022-test.jsonl')

    parser.add_argument('--max_seq_length', type=int, default= 512)
    args= parser.parse_args()

    return args

def get_num_classes(args):
  if args.label_columns== 'label_aspect_1':
    args.num_classes= 3
  elif args.label_columns== 'label_aspect_2':
    args.num_classes= len(STAGE2_ASPECT)
  elif args.label_columns== 'label_sentiment_1':
    args.num_classes= 2
  elif args.label_columns== 'label_sentiment_2':
    args.num_classes= 2

  return args

if __name__ == '__main__':
  args= get_config()
  args.device= 'cuda' if torch.cuda.is_available() else 'cpu'
  seed_everything(args.seed)

  prediction_results= defaultdict(list)

  LABEL_DICT= [stage1_num_to_sentiment_dict, stage2_num_to_sentiment_dict, stage1_num_to_aspect_dict, stage2_num_to_aspect_dict]
  LABEL_ORDER= ['label_sentiment_1', 'label_sentiment_2', 'label_aspect_1', 'label_aspect_2']
  MODEL_PATH= ['./results_sentiment_1/checkpoint-298', './results_sentiment_2/checkpoint-60', './results_aspect_1/checkpoint-3663', './results_aspect_2/checkpoint-1368', ]
  
  # model= AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)
  tokenizer= AutoTokenizer.from_pretrained(args.model_name_or_path)

  idx= 0
  while idx <4:
    
    args.label_columns= LABEL_ORDER[idx]
    args= get_num_classes(args)
    config= AutoConfig.from_pretrained(args.model_name_or_path)
    config.num_labels= args.num_classes

    model= AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config= config)
    print(model.config.num_labels)
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH[idx], 'pytorch_model.bin')))
    model.to(args.device)
    

    with open(args.test_data_path, 'r') as f:
      for i, val in tqdm(enumerate(f)):
        example= json.loads(val)['sentence_form']
        tokenized_example= tokenizer(
          example,
          truncation= True,
          padding= 'max_length',
          max_length= args.max_seq_length,
          return_token_type_ids= False,
          return_tensors= 'pt'
        )

        outputs= model(
          input_ids= tokenized_example['input_ids'].to(args.device),
          attention_mask= tokenized_example['attention_mask'].to(args.device)
        ).logits
        preds= outputs.argmax(dim= -1)
        word_preds= LABEL_DICT[idx][int(preds[0])]

        prediction_results[LABEL_ORDER[idx]].append(word_preds)      
    idx+=1

  submission= []
  with open(args.test_data_path, 'r') as f:
    for i, val in tqdm(enumerate(f)):
      example= json.loads(val)
      annot= []

      if prediction_results['label_aspect_1'][i]== '본품#품질':
        annot.append(prediction_results['label_aspect_1'][i])
      elif prediction_results['label_aspect_1'][i]== '제품 전체#일반':
        annot.append(prediction_results['label_aspect_1'][i])
      else:
        annot.append(prediction_results['label_aspect_2'][i])
      
      if prediction_results['label_sentiment_1'][i]== 'positive':
        annot.append(prediction_results['label_sentiment_1'][i])
      else:
        annot.append(prediction_results['label_sentiment_2'][i])
      
      example['annotation']= [annot]

      # if i==10: break;

      submission.append(example)

  with open('./submission.jsonl', 'w') as f:
    for i, val in tqdm(enumerate(submission)):
      f.write(json.dumps(val, ensure_ascii=False) + "\n") 

        
      


  



      
      










      


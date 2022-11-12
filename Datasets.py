from torch.utils.data import Dataset
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
import jsonlines
import json
class Custom_Dataset(Dataset):
    def __init__(self, dataset_path, mode, type_path, tokenizer, input_length):
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.type_path = type_path
        self.mode = mode
        if mode == 'train' or mode == 'valid':
            self.dataset_path = dataset_path
            self.dataset = pd.read_csv(dataset_path, encoding='utf-8')
            print(f'Getting dataset {dataset_path} with length {len(self.dataset)}')
        elif mode == 'kfold':
            self.dataset = dataset_path
        elif mode == 'eval':
            self.dataset = dataset_path
            print(f'Getting dataset eval_topic with length {len(self.dataset)}')
        else:
            raise Exception("Wrong mode")
        
    def __len__(self):
        return len(self.dataset)

    def convert_to_features(self, example_batch, index=None):  
        if self.mode == 'train' or self.mode == 'valid' or self.mode == 'kfold':
            if self.type_path == 'category' or self.type_path == 'trinary':
                input_, label = example_batch['input'], example_batch['label']
                label = torch.tensor(json.loads(label))
            elif self.type_path == 'topic':
                input_, entity_, label = example_batch['input'], example_batch['entity'], example_batch['label']
                label = torch.tensor(json.loads(label))
            elif self.type_path == "sentiment":
                input_, entity_, label = example_batch['input'], example_batch['entity'], example_batch['label']
        elif self.mode == 'eval':
            if self.type_path == 'category' or self.type_path == 'trinary':
                input_, entity_, label = example_batch[0], [], []
            else:
                input_, entity_, label = example_batch[0], example_batch[1], []
                
        if self.type_path == 'category':
            context = "다음 문장의 카테고리를 분류하면?" + "[SEP]" + input_ #ver2
        elif self.type_path == 'topic':
            context = f"{entity_}에 대해 [제품 전체, 본품, 패키지/구성품, 브랜드] 중에 하나를 고르면?" + "[SEP]" + input_
        elif self.type_path == 'sentiment':
            context = f"{entity_.split('#')[0]}의 {entity_.split('#')[1]}에 대한 감성을 분류하면?" + "[SEP]" + input_
        elif self.type_path == 'trinary':
            context = "다음 문장은 품질, 일반, 나머지 중에 뭐에 대한 거야?" + "[SEP]" + input_ 

        source = self.tokenizer.batch_encode_plus([str(context)], max_length=self.input_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt", return_token_type_ids=True)

        return source, label

    def __getitem__(self, index):
        if self.mode == 'train' or self.mode == 'valid' or self.mode == 'kfold':
            source, label = self.convert_to_features(self.dataset.iloc[index], index=index) 
        elif self.mode == 'eval':
            source, label = self.convert_to_features(self.dataset[index], index=index)

        source_ids = source["input_ids"].squeeze()
        src_mask = source["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "labels": label}

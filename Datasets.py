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
            if self.type_path == 'category':
                input_, label = example_batch['input'], example_batch['label']
                #input_, entity_, label = example_batch['input'], example_batch['entity'], example_batch['label']
                label = torch.tensor(json.loads(label))
            elif self.type_path == 'topic' or self.type_path == 'topic_trinary':
                input_, entity_, label = example_batch['input'], example_batch['entity'], example_batch['label']
                #input_, label = example_batch['input'], example_batch['entity'], example_batch['label']
                label = torch.tensor(json.loads(label))
            elif self.type_path == 'trinary':
                input_, label = example_batch['input'], example_batch['label']
                label = torch.tensor(json.loads(label))
            elif self.type_path == "sentiment" or self.type_path == 'binary':
                input_, entity_, label = example_batch['input'], example_batch['entity'], example_batch['label']
        elif self.mode == 'eval':
            if self.type_path == 'category' or self.type_path == 'trinary':
                input_, entity_, label = example_batch[0], [], []
            elif self.type_path == 'topic' or self.type_path == 'sentiment' or self.type_path == 'topic_binary' or self.type_path == 'category_binary':
                input_, entity_, label = example_batch[0], example_batch[1], []
                
        if self.type_path == 'category':
            context = "다음 문장의 카테고리를 분류하면?" + "[SEP]" + input_ #ver2
            #context = "편의성, 디자인, 인지도, 가격, 다양성 중에 고르면?" + "[SEP]" + input_ #ver2
            #context = "문장을 편의성 / 디자인 / 인지도 / 가격 / 다양성 중에 하나로 분류하면?" + "[SEP]" + input_ 
        elif self.type_path == 'topic' or self.type_path == 'topic_binary' or self.type_path == 'topic_trinary':
            #context = entity_ + "가 카테고리일때 토픽이 뭐야?" + "[SEP]" + input_ #ver2
            context = f"{entity_}에 대해 [제품 전체, 본품, 패키지/구성품, 브랜드] 중에 하나를 고르면?" + "[SEP]" + input_
            #context = f"[제품 전체, 본품, 패키지/구성품, 브랜드] 중에 {entity_}에 대해 고르세요." + "[SEP]" + input_
            #context = f"문장의 주제를 분류해봐!" + "[SEP]" + input_
            #context = f"다음 문장의 주제를 분류하면?" + "[SEP]" + input_
            #context = f"[제품 전체, 본품, 패키지/구성품, 브랜드] 중에 고르시오. {entity_}에 대한 것이다." + "[SEP]" + input_
        elif self.type_path == 'topic_binary':
            context = f"다음 글의 주제는 {entity_}가 맞아?" + "[SEP]" + input_
        elif self.type_path == 'binary' or self.type_path == 'category_binary':
            context = f"{entity_.split('#')[0]}의 카테고리는 {entity_.split('#')[1]}이 맞아?" + "[SEP]" + input_
        elif self.type_path == 'sentiment':
            context = f"{entity_.split('#')[0]}의 {entity_.split('#')[1]}에 대한 감성을 분류하면?" + "[SEP]" + input_
            #context = f"문장에서 {entity_.split('#')[1]}의 감성은 뭐야?" + "[SEP]" + input_
            #context = f"긍정, 부정, 중립 중에 {entity_.split('#')[0]}의 {entity_.split('#')[1]}에 대한 건 뭐야?"  + "[SEP]" + input_
            #context = f"[긍정, 부정, 중립] 중에 고르시오. {entity_.split('#')[0]}의 카테고리중에 {entity_.split('#')[1]}에 대한 것이다."  + "[SEP]" + input_
        elif self.type_path == 'trinary':
            context = "다음 문장은 품질, 일반, 나머지 중에 뭐에 대한 거야?" + "[SEP]" + input_ #ver2
            #context = "선택지: [품질], [일반], [나머지]" + "[SEP]" + input_ #ver2
            #context = "품질은 제품의 상태가 좋은지이고, 일반은 상품 전체의 느낌이야. 아니면 나머지고. 품질, 일반, 나머지 중에 하나를 고르면?" + "[SEP]" + input_ #ver2
            #context = "다음 문장에서 특징을 분류해봐!" + "[SEP]" + input_ #ver2
            #context = "문장은 뭐에 대한 거야?" + "[SEP]" + input_ #ver3
            #context = "다음 문장의 카테고리를 분류하면?" + "[SEP]" + input_
            #context = "[품질, 일반, 기타] 중에서 하나로 분류하면?" +"[SEP]" + input_

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

class Custom_Dataset_GPT2_Eval(Dataset):
    def __init__(self, dataset_name, tokenizer, input_list, input_length, output_length, args):
        self.input_length = input_length
        self.output_length = output_length
        self.tokenizer = tokenizer
        self.dataset_path = dataset_name

        with jsonlines.open(f'{self.root_path}/{self.dataset_path}') as f:
            for line in f:
                self.dataset.append(line)
        print(f'Getting dataset {self.dataset_path} with length {len(self.dataset)}')
        
    def __len__(self):
        return len(self.dataset)

    def convert_to_features(self, example_batch, index=None):  

        input_, target_ = example_batch['context'] + "<s>", example_batch['answer']
        source = self.tokenizer.batch_encode_plus([str(input_)], max_length=self.input_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt")
        targets = self.tokenizer.batch_encode_plus([str(target_)], max_length=self.output_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt") 
        return source, targets

    def __getitem__(self, index):
        source, targets = self.convert_to_features(self.dataset[index])
        
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}
import pytorch_lightning as pl
from transformers import (
    Adafactor, 
    BertTokenizer, 
    BertForSequenceClassification,
    XLMRobertaModel, 
    AutoTokenizer, 
    BertModel, 
    FunnelTokenizerFast, 
    FunnelModel,
    ElectraTokenizerFast, 
    BertTokenizerFast,
    ElectraModel,
)
import torch
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn import functional as F
import os
import pandas as pd
from torch import nn

import deepspeed
import numpy as np
from Datasets import Custom_Dataset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
#from kobert_transformers import get_tokenizer
from datasets import concatenate_datasets

import pandas as pd


class BERT(pl.LightningModule):
    def __init__(self, hparams, mode):
        super(BERT, self).__init__()    

        self.mode = mode
        if self.mode == 'sentiment':
            self.num_label = 3
        elif self.mode == 'trinary':
            self.num_label = 3
        elif self.mode == 'category':
            self.num_label = 5
        elif self.mode == 'topic':
            self.num_label = 4
        else:
            raise Exception("Wrong mode")

        self.train_path = hparams.train_path
        self.eval_path = hparams.eval_path
        self.tokenizer = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")
        self.model = BertModel.from_pretrained("kykim/bert-kor-base")

        self.fold_option= hparams.fold_option
        self.train_idx= hparams.train_idx
        self.valid_idx= hparams.valid_idx
        
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.labels_classifier = nn.Linear(768, self.num_label)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()
        if self.mode == 'sentiment':
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.BCELoss()

        self.save_hyperparameters(hparams)

        self.train_pred = []
        self.train_gt = []
        self.val_pred = []
        self.val_gt = []
        self.epoch_num = 0
        self.threshold = 0.5

        self.total_dataset= self.get_total_dataset()
        self.log('total dataset', len(self.total_dataset))


    def forward(self, input_ids, input_mask, labels=None):
        sep_idx = []
        print(input_ids)
        for i in range(len(input_ids)):
            if input_ids[i] == 3:
                sep_idx.append(i)
        token_type_ids = torch.zeros_like(input_ids)
        print(input_ids)
        print(sep_idx)
        for i in range(len(token_type_ids)):
            if i > sep_idx[0] and i <= sep_idx[1]:
                token_type_ids[i] = 1
        print(token_type_ids)
        exit()
        output = self.model(
            input_ids=input_ids,
            attention_mask=input_mask
        )
        output = self.dropout(output.pooler_output)
        output = self.labels_classifier(output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            if self.mode == 'sentiment':
                loss = self.criterion(output, labels)
            else:
                loss = self.criterion(output.to(torch.float16), labels.to(torch.float16))

        return loss, output


    def training_step(self, batch):
        loss, outputs = self(batch['source_ids'], batch['source_mask'], batch['labels'])

        if self.mode == 'sentiment':
            self.train_pred.append(torch.argmax(outputs).detach().cpu())
        elif self.mode == 'topic' or self.mode == 'trinary' or self.mode == 'category':
            self.train_pred.append(outputs[0].detach().cpu())

        self.train_gt.append(batch['labels'].detach().cpu()[0])
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        if self.mode == 'sentiment':
            acc = accuracy_score(self.train_gt, self.train_pred)
        elif self.mode == 'topic' or self.mode == 'trinary' or self.mode == 'category':
            self.train_pred = torch.stack(self.train_pred)
            self.train_gt = torch.stack(self.train_gt)
            self.train_pred = torch.where(self.train_pred > self.threshold, 1, 0)
            acc = accuracy_score(self.train_gt, self.train_pred)
        f1 = f1_score(self.train_pred, self.train_gt, average='macro')

        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.train_pred = []
        self.train_gt = []
        model_saved_path = self.hparams.checkpoint_path + self.mode + str(self.epoch_num) + '.pt'
        param_dict = {}
        for name, param in self.model.state_dict().items():
            param_dict[name]=param.clone().detach().cpu()
        for name, param in self.labels_classifier.state_dict().items():
            param_dict["label_classifier."+name] = param.clone().detach().cpu()

        torch.save(param_dict, model_saved_path) 

        self.epoch_num += 1

    def validation_step(self, batch, batch_idx):
        loss, outputs = self(batch['source_ids'], batch['source_mask'], batch['labels'])
        if self.mode == 'sentiment':
            self.val_pred.append(torch.argmax(outputs).detach().cpu())
        elif self.mode == 'topic' or self.mode == 'trinary' or self.mode == 'category':
            self.val_pred.append(outputs[0].detach().cpu())

        self.val_gt.append(batch['labels'].detach().cpu()[0])

        return loss

    def on_validation_epoch_end(self):
        if self.mode == 'sentiment':
            acc = accuracy_score(self.val_gt, self.val_pred)
        elif self.mode == 'topic' or self.mode == 'trinary' or self.mode == 'category':
            self.val_pred = torch.stack(self.val_pred)
            self.val_gt = torch.stack(self.val_gt)
            self.val_pred = torch.where(self.val_pred > self.threshold, 1, 0)
            acc = accuracy_score(self.val_gt, self.val_pred)
        f1_all = f1_score(self.val_pred, self.val_gt, average=None)
        f1_macro = f1_score(self.val_pred, self.val_gt, average='macro')
        f1_micro = f1_score(self.val_pred, self.val_gt, average='micro')
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for f1 in range(len(f1_all)):
            self.log(f'val_f1_class_{f1}', f1_all[f1], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1_macro', f1_macro, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1_micro', f1_micro, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        if self.mode == 'sentiment':
            count_list = [0,0,0]
            for i in self.val_pred:
                count_list[int(i)] += 1
            all_sum = sum(count_list)
            self.log('val_pos_per', count_list[0]/all_sum * 100, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_neg_per', count_list[1]/all_sum * 100, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_neu_per', count_list[2]/all_sum * 100, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.val_pred = []
        self.val_gt = []


    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        model = self.model
        
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        parameters = model.parameters()

        if self.hparams.accelerator=='deepspeed_stage_2':
            optimizer = deepspeed.ops.adam.FusedAdam(parameters, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay, betas=(0.9, 0.95))
        elif self.hparams.accelerator=='deepspeed_stage_2_offload':
            optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(parameters, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay, betas=(0.9, 0.95))
        else:
            optimizer = Adafactor(optimizer_grouped_parameters, lr=self.hparams.learning_rate, scale_parameter=False, relative_step=False)
            #optimizer = optim.Adam(parameters, lr=self.hparams.learning_rate)

        return [optimizer]

    def train_dataloader(self):
        train_dataset = Custom_Dataset(self.train_path, "train", self.mode, self.tokenizer, self.hparams.input_length)
        sampler = RandomSampler(train_dataset)
        dataloader = DataLoader(train_dataset,  sampler=sampler,batch_size=self.hparams.train_batch_size, drop_last=True, num_workers=self.hparams.num_workers)
        return dataloader

    def val_dataloader(self):
        validation_dataset = Custom_Dataset(self.eval_path, "valid", self.mode, self.tokenizer, self.hparams.input_length)
        return DataLoader(validation_dataset, batch_size=self.hparams.eval_batch_size, num_workers=self.hparams.num_workers, shuffle=False) 
    
    def get_total_dataset(self):
        
        train_df = Custom_Dataset(self.train_path, "train", self.mode, self.tokenizer, self.hparams.input_length).dataset
        valid_df = Custom_Dataset(self.eval_path, "valid", self.mode, self.tokenizer, self.hparams.input_length).dataset
        
        self.total_df= pd.concat([train_df, valid_df])
        return self.total_df
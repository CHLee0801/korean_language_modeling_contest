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
    AutoModel
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
from Datasets import Custom_Dataset_Binary
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
#from kobert_transformers import get_tokenizer

class ROBERTA_BINARY(pl.LightningModule):
    def __init__(self, hparams, mode):
        super(ROBERTA_BINARY, self).__init__()    

        self.num_label = 2
        self.train_path = hparams.train_path
        self.eval_path = hparams.eval_path
        self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
        self.model = AutoModel.from_pretrained("klue/roberta-large")
        
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.labels_classifier = nn.Linear(1024, self.num_label)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters(hparams)

        self.train_pred = []
        self.train_gt = []
        self.val_pred = []
        self.val_gt = []
        self.epoch_num = 0

    def forward(self, input_ids, input_mask, labels=None):
        output = self.model(
            input_ids=input_ids,
            attention_mask=input_mask
        )
        output = self.labels_classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output


    def training_step(self, batch):
        loss, outputs = self(batch['source_ids'], batch['source_mask'], batch['labels'])

        self.train_pred.append(torch.argmax(outputs).detach().cpu())
        self.train_gt.append(batch['labels'].detach().cpu()[0])
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):

        acc = accuracy_score(self.train_gt, self.train_pred)
        f1 = f1_score(self.train_gt, self.train_pred, average='macro')
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.train_pred = []
        self.train_gt = []
        model_saved_path = self.hparams.checkpoint_path + 'epoch_' + str(self.epoch_num) + '.pt'
        param_dict = {}
        for name, param in self.model.state_dict().items():
            param_dict[name]=param.clone().detach().cpu()
        for name, param in self.labels_classifier.state_dict().items():
            param_dict["label_classifier."+name] = param.clone().detach().cpu()

        #torch.save(param_dict, model_saved_path) 

        self.epoch_num += 1

    def validation_step(self, batch, batch_idx):
        loss, outputs = self(batch['source_ids'], batch['source_mask'], batch['labels'])
        self.val_pred.append(torch.argmax(outputs).detach().cpu())
        self.val_gt.append(batch['labels'].detach().cpu()[0])
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def on_validation_epoch_end(self):

        acc = accuracy_score(self.val_gt, self.val_pred)
        f1 = f1_score(self.val_gt, self.val_pred)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
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
        train_dataset = Custom_Dataset_Binary(self.train_path, "train", self.tokenizer, self.hparams.input_length)
        sampler = RandomSampler(train_dataset)
        dataloader = DataLoader(train_dataset,  sampler=sampler,batch_size=self.hparams.train_batch_size, drop_last=True, num_workers=self.hparams.num_workers)
        return dataloader

    def val_dataloader(self):
        validation_dataset = Custom_Dataset_Binary(self.eval_path, "valid", self.tokenizer, self.hparams.input_length)
        return DataLoader(validation_dataset, batch_size=self.hparams.eval_batch_size, num_workers=self.hparams.num_workers, shuffle=False)
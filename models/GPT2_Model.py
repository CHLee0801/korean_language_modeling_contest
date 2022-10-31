import pytorch_lightning as pl
#from models.Modular_GPT2 import GPT2LMHeadModel as GPT2_Modular
#from models.Kadapter_GPT2 import GPT2LMHeadModel as GPT2_Kadapter
#from models.Lora_GPT2 import GPT2LMHeadModel as GPT2_Lora
#from models.RecAdam import RecAdam

from transformers import (
    Adafactor,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    PreTrainedTokenizerFast, 
    AutoTokenizer, 
    AutoModelForCausalLM 
)

import torch
from Datasets import Custom_Dataset_GPT2
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader, ConcatDataset

import re
import string
import os
class GPT2(pl.LightningModule):
    def __init__(self, hparams):
        super(GPT2, self).__init__()
        self.save_hyperparameters(hparams)      

        if not os.path.exists(hparams.checkpoint_path):
            os.makedirs(hparams.checkpoint_path)
        self.mix_ratio = 4
        self.mix_decay = 0.7
        self.epoch = 0
        self.tokenizer = AutoTokenizer.from_pretrained(
            'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
            bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
            pad_token_id=self.tokenizer.eos_token_id,
            torch_dtype='auto'
        )
        


        if hparams.freeze_level==0: # Do not freeze any parameters
            print('Not freezing any parameters!')
        elif hparams.freeze_level==1: # Freeze model
            self.freeze_params(self.model) 

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.tokenizer.padding_side = "left"

        self.output_dir = self.hparams.output_dir

        
    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()
        
        def rid_of_specials(text):
            text = text.replace("<extra_id_0>", "")
            text = text.replace("<extra_id_1>", "")
            return text

        return rid_of_specials(white_space_fix(remove_articles(remove_punc(lower(s)))))

    def exact_match_score(self, prediction, ground_truth):
        if self.normalize_answer(ground_truth) in self.normalize_answer((prediction)):
            return 1
        else:
            return 0
        #return int(self.normalize_answer(prediction) == self.normalize_answer(ground_truth))

    def calculate_scores(self, predictions, ground_truths):
        em_score = 0
        
        for i in range(len(predictions)):
            ground_truth = ground_truths[i]
            prediction = predictions[i]
            em_score +=  self.exact_match_score(prediction, ground_truth)
        
        em_score /= len(predictions)
        return em_score*100

    def get_dataset(self, dataset_name, tokenizer, type_path, args, length=None):
        dataset = Custom_Dataset_GPT2(dataset_name=dataset_name, tokenizer=tokenizer, type_path=type_path, input_length=args.input_length, 
                    output_length=args.output_length, args=args)
        return dataset


    def freeze_params(self, model):
        for par in model.parameters():
            par.requires_grad = False
    
    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x))

    def is_logger(self):
        return self.trainer.global_rank <= 0
    
    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=lm_labels,
    )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
        )

        loss = outputs[0]
        return loss
    
    
    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, gen_text)
    
     
    def _generative_step(self, batch, batch_idx, dataloader_idx = -1):
        
        input_length = self.hparams.input_length
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                batch["source_ids"],
                attention_mask=batch["source_mask"],
                use_cache=True,
                max_length=self.hparams.input_length + 5,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                num_beams=2,
                early_stopping=True
            )

        generated_ids = torch.transpose(torch.transpose(generated_ids,0,1)[input_length:],0,1)
        preds = self.ids_to_clean_text(generated_ids)
        clean_preds = []
        for text in preds:
            if "." in text:
                clean_preds.append(text[:text.find(".")+1])
            else: 
                clean_preds.append(text)

        print("clean_preds",clean_preds)
        targets = self.ids_to_clean_text(batch["target_ids"])
        print("targets",targets)
    
        loss = self._step(batch)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        em_score = self.calculate_scores(clean_preds, targets)
        em_score = torch.tensor(em_score,dtype=torch.float32)
        if dataloader_idx == 0:
            self.log('sentiment_em_score', em_score, prog_bar=True, logger=True)
        else:
            self.log('topic_em_score', em_score, prog_bar=True, logger=True)
    
    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def on_train_epoch_end(self):
        model_saved_path = self.hparams.checkpoint_path + str(self.epoch) + '.pt'
        if self.epoch % self.hparams.check_val_every_n_epoch == 4:
            torch.save(self.model.state_dict(), model_saved_path)
        self.epoch+=1

    def validation_step(self, batch, batch_idx, dataloader_idx = -1):
        return self._generative_step(batch, batch_idx, dataloader_idx)

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
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
        optimizer = Adafactor(optimizer_grouped_parameters, lr=self.hparams.learning_rate, scale_parameter=False, relative_step=False)

        self.optimizer = optimizer
        len_data = len(self.train_dataloader())
        denomniator = self.hparams.n_gpu * self.hparams.gradient_accumulation_steps
        steps_per_epoch = ( len_data // denomniator ) + 1
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.learning_rate, steps_per_epoch=steps_per_epoch, pct_start=0.1, epochs=self.hparams.num_train_epochs, anneal_strategy='linear', cycle_momentum=False)

        if self.hparams.use_lr_scheduling:
            return [optimizer], [{"scheduler": lr_scheduler, "interval": "step", "name": "learning rate"}]
        else:
            return [optimizer]

    def train_dataloader(self):   
        if self.epoch < 10:
            epoch_str = f"0{self.epoch}"
        else:
            epoch_str = str(self.epoch)
        train_dataset = self.get_dataset(dataset_name = f"train_{epoch_str}.jsonl",tokenizer=self.tokenizer, type_path="train", args=self.hparams)
        sampler=RandomSampler(train_dataset)
        dataloader = DataLoader(train_dataset, sampler=sampler,  batch_size=self.hparams.train_batch_size, drop_last=True, num_workers=self.hparams.num_workers)
        return dataloader

    def val_dataloader(self):
        val_0 = self.get_dataset(dataset_name = 'dev_sentiment.jsonl', tokenizer=self.tokenizer, type_path="validation", args=self.hparams)
        val_1 = self.get_dataset(dataset_name = 'dev_topic.jsonl', tokenizer=self.tokenizer, type_path="validation", args=self.hparams)
        return [DataLoader(val_0, batch_size=self.hparams.eval_batch_size, num_workers=self.hparams.num_workers, shuffle=False), 
            DataLoader(val_1, batch_size=self.hparams.eval_batch_size, num_workers=self.hparams.num_workers, shuffle=False)
        ]

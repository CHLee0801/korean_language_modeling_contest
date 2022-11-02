import os
import time
import argparse
from argparse import ArgumentParser
import os
import json
import random
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.plugins import DeepSpeedPlugin
from Evaluation_2 import evaluate

from sklearn.model_selection import train_test_split, StratifiedKFold



from models import load_model
CUDA_LAUNCH_BLOCKING=1
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    #Parsing Arguments
    parser = ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    arg_ = parser.parse_args()
    if arg_.config == None:
        raise NameError("Include a config file in the argument please.")
    #Getting configurations
    config_path = arg_.config
    with open(config_path) as config_file:
        hparam = json.load(config_file)
    hparam = argparse.Namespace(**hparam)

    #Setting GPUs to use
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=hparam.CUDA_VISIBLE_DEVICES

    #Init configs that are not given
    if 'grad_norm' not in hparam:
        hparam.grad_norm = 0.1
    if 'weight_decay' not in hparam:
        hparam.weight_decay = 0.01
    if 'output_log' not in hparam:
        hparam.output_log = None
    if 'pred_log' not in hparam:
        hparam.pred_log = None
    if 'num_files' not in hparam:
        hparam.num_files = 1
    if 'learning_rate' not in hparam:
        hparam.learning_rate = None
    if 'gradient_accumulation_steps' not in hparam:
        hparam.gradient_accumulation_steps = 1
    if 'num_train_epochs' not in hparam:
        hparam.num_train_epochs = 0
    if 'use_lr_scheduling' not in hparam:
        hparam.use_lr_scheduling = False
    if 'num_workers' not in hparam:
        hparam.num_workers = 0
    if 'output_dir' not in hparam:
        hparam.output_dir = None
    if 'wandb_log' not in hparam:
        hparam.wandb_log = False
    if 'accelerator' not in hparam:
        hparam.accelerator = None
    if 'fp16' not in hparam:
        hparam.fp16 = False
    if 'train_path' not in hparam:
        hparam.train_path = None
    if "eval_path" not in hparam:
        hparam.eval_path = None
    if "val_check_interval" not in hparam:
        hparam.val_check_interval = 1.0
    if "freeze_level" not in hparam:
        hparam.freeze_level = 0
    if "checkpoint_path" not in hparam:
        hparam.checkpoint_path = ""
    if "checkpoint_path_1" not in hparam:
        hparam.checkpoint_path_1 = ""
    if "checkpoint_path_2" not in hparam:
        hparam.checkpoint_path_2 = ""
    if "checkpoint_path_3" not in hparam:
        hparam.checkpoint_path_3 = ""
    if "checkpoint_path_4" not in hparam:
        hparam.checkpoint_path_4 = ""
    if "checkpoint_path_5" not in hparam:
        hparam.checkpoint_path_5 = ""
    if "test" not in hparam:
        hparam.test = False
    if "model" not in hparam:
        hparam.model = ""
    if "mode" not in hparam:
        hparam.mode = ""
    #Logging into WANDB if needed
    if hparam.wandb_log:
        wandb_logger = WandbLogger(project=hparam.wandb_project, name=hparam.wandb_run_name, entity="changholee")
    else:
        wandb_logger = None
        
    #Setting configurations
    args_dict = dict(
        output_dir=hparam.output_dir, # Path to save the checkpoints
        train_path=hparam.train_path,
        eval_path=hparam.eval_path,
        num_files = hparam.num_files,
        learning_rate=hparam.learning_rate,
        adam_epsilon=1e-8,
        train_batch_size=hparam.train_batch_size,
        eval_batch_size=hparam.eval_batch_size,
        weight_decay=hparam.weight_decay,
        num_train_epochs=hparam.num_train_epochs,
        gradient_accumulation_steps=hparam.gradient_accumulation_steps,
        n_gpu=hparam.ngpu,
        num_workers=hparam.num_workers,
        use_lr_scheduling = hparam.use_lr_scheduling,
        val_check_interval = hparam.val_check_interval,
        fp16=hparam.fp16,
        opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        max_grad_norm=hparam.grad_norm, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
        seed=42,
        accelerator=hparam.accelerator,
        output_log=hparam.output_log,
        pred_log = hparam.pred_log,
        checkpoint_path = hparam.checkpoint_path,
        check_val_every_n_epoch=hparam.check_val_every_n_epoch,
        CUDA_VISIBLE_DEVICES = hparam.CUDA_VISIBLE_DEVICES,
        mode = hparam.mode,
        input_length = hparam.input_length,
        output_length = hparam.output_length,
        freeze_level = hparam.freeze_level,
        check_validation = hparam.check_validation,
        checkpoint_path_1 = hparam.checkpoint_path_1,
        checkpoint_path_2 = hparam.checkpoint_path_2,
        checkpoint_path_3 = hparam.checkpoint_path_3,
        checkpoint_path_4 = hparam.checkpoint_path_4,
        checkpoint_path_5 = hparam.checkpoint_path_5,
        test = hparam.test,
        model = hparam.model
    )
    args = argparse.Namespace(**args_dict)

    if args.output_dir != "" and not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path, exist_ok=True)

    checkpoint_callback = False # Do not save model checkpoints when output dir is empty
    callbacks=[]
    
    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        max_epochs=int(args.num_train_epochs * args.num_files),
        precision= 16 if args.fp16 else 32,
        amp_backend="native",
        gradient_clip_val=args.max_grad_norm,
        enable_checkpointing=checkpoint_callback,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        #val_check_interval=args.val_check_interval,
        #default_root_dir=args.output_dir,
        logger = wandb_logger,
        callbacks = callbacks,
        strategy = args.accelerator,
    )

    
    if args.check_validation == True:
        Model_1 = load_model('bert')
        Model_2 = load_model('roberta')
        Model_3 = load_model('electra')
        evaluate(args, Model_1, Model_2, Model_3)
    else:
        if 'bert' == args.model:
            Model = load_model('bert')
        elif 'roberta' == args.model:
            Model = load_model('roberta')
        elif 'electra' == args.model:
            Model = load_model('electra')
        else:
            raise Exception('currently not supporting given model')
        

        set_seed(40) # very important to set random seed since we mix training data during training. requires for DDP. 
        model = Model(args, args.mode)
        trainer = pl.Trainer(**train_params)
        trainer.fit(model)

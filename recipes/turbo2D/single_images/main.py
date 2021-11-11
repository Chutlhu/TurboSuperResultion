import argparse

import os
import numpy as np

# DNN Libraries
# import pl torch libs
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping

# Turboflow Libraries
from turboflow.dataloaders import TurboFlowDataModule
from turboflow.models.phyrff_hard import plDivFreeRFFNet

# os.environ['WANDB_CONSOLE'] = 'off' # shutdown "AssertionError: can only test a child process"

def get_path_and_prepare_folder():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    return root_dir

def get_args():

    # these are project-wide arguments
    parser = argparse.ArgumentParser(prog='main.py', add_help=True)
    
    # add data specific arguments
    parser = TurboFlowDataModule.add_data_specific_args(parser)
    parser = plDivFreeRFFNet.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    arg_groups={}

    for group in parser._action_groups:
        group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
        arg_groups[group.title]=argparse.Namespace(**group_dict)
    
    return arg_groups


if __name__ == '__main__':
    
    hparams = get_args()

    # PREPARE FOLDER
    root_dir = get_path_and_prepare_folder()

    dm = TurboFlowDataModule.from_argparse_args(hparams['data'])

    # MODEL
    model = plDivFreeRFFNet(**vars(hparams['model']))

    early_stop_callback = EarlyStopping(monitor='val_loss')
    trainer = Trainer.from_argparse_args(hparams['pl.Trainer'], 
            callbacks=[early_stop_callback])

    # TRAIN    
    trainer.fit(model, dm)
    
    # TEST
    if not hparams['pl.Trainer'].fast_dev_run:
        trainer.test()
import argparse

import os
import numpy as np

# DNN Libraries
import torch
# import pl torch libs
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# Turboflow Libraries
from turboflow.dataloaders import Turbo2DDataModule
from turboflow.models.phyrff_hard import plDivFreeRFFNet

def get_path_and_prepare_folder():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    return root_dir

def get_args():

    # these are project-wide arguments
    parser = argparse.ArgumentParser(prog='main.py', add_help=True)
    parser.add_argument('--fast_dev_run',
                               default=False, action='store_true',
                               help='fast_dev_run: runs 1 batch of train, test, val (ie: a unit test)')
    # add data specific arguments
    parser = Turbo2DDataModule.add_data_specific_args(parser)
    # add model specific arguments
    parser = plDivFreeRFFNet.add_model_specific_args(parser)
    # add all the available trainer options to argparse
    return parser.parse_args()


if __name__ == '__main__':
    
    hparams = get_args()
    print(hparams)

    # PREPARE FOLDER
    root_dir = get_path_and_prepare_folder()

    # # DATA
    dm = Turbo2DDataModule(hparams)

    # MODEL
    model = plDivFreeRFFNet(hparams)

    early_stop_callback = EarlyStopping(monitor='val_loss')
    trainer = Trainer(gpus=1, 
                     fast_dev_run=hparams.fast_dev_run,
                     log_every_n_steps=1,
                     check_val_every_n_epoch=200, 
                     max_epochs=5000, 
                     callbacks=[early_stop_callback])

    # TRAIN    
    trainer.fit(model, dm)
    
    # TEST
    if not hparams.fast_dev_run:
        trainer.test()
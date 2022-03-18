import gc

from argparse import Namespace
from mimetypes import suffix_map
from pathlib import Path
import numpy as np
import yaml

from pytorch_lightning.trainer import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from turboflow.dataloaders import TurboFlowDataModule
from turboflow.models.phyrff import plDivFreeRFFNet

import matplotlib.pyplot as plt

def train(exp_suffix, data_dir, hparams, train_params):


    seed = train_params['seed']
    dt = train_params['dt']
    dx = train_params['dx']
    nt_train = train_params['nt_train']
    batch_size = train_params['batch_size']
    n_epoch = train_params['n_epoch']

    seed_everything(seed, workers=True)

    # load dataset for TRAIN
    dm_train = TurboFlowDataModule(
        dataset='Turb2D', 
        data_dir=data_dir,
        time_idx=np.arange(dt['train']*nt_train),
        
        train_batch_size=batch_size['train'],
        val_batch_size=batch_size['val'],
        test_batch_size=128,

        train_downsampling_space=dx['train'],
        val_downsampling_space=dx['val'],
        test_downsampling_space=64,
        
        train_downsampling_time=dt['train'],
        val_downsampling_time=dt['val'],
        test_downsampling_time=16,
        
        train_shuffle=True,
        val_shuffle=False,
        test_shuffle=False,
        num_workers=8)

    dm_train.setup(stage='fit')

    datasets = [dm_train.train_dataset, dm_train.val_dataset]

    for dataset in datasets:
        X, y = dataset[:]
        print(X.shape, y.shape, dataset.img_res, dataset.vars_shape_img)

    model = plDivFreeRFFNet(**vars(Namespace(**hparams)))


    early_stop_callback = EarlyStopping(
        monitor='val/loss/tot', 
        patience=3,
        min_delta=1e-5)
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss/tot",
        dirpath=".torch_checkpoints",
        filename="Turb2D-%s-%s-{epoch:02d}-{val_loss:.2f}" % (hparams['name'], exp_suffix),
        save_top_k=1,
        mode="min",
    )

    trainer = Trainer(
        gpus=1,
        max_epochs=n_epoch, 
        log_every_n_steps=5,
        check_val_every_n_epoch=10, 
        callbacks=[early_stop_callback, checkpoint_callback])
    # print(model)

    trainer.fit(model, dm_train)

    print('Path to best model', checkpoint_callback.best_model_path)
    best_model_path = checkpoint_callback.best_model_path
    
    del dm_train
    del dataset
    del trainer

    collected = gc.collect()
    print("Garbage collector: collected %d objects." % collected)

    return model, best_model_path, hparams
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

base_dir = Path('/','home','dicarlo_d','Documents','Code','TurboSuperResultion','recipes','time_cnn')
data_dir = Path('/','home','dicarlo_d','Documents','Datasets','Turb2D.hdf5')
fig_path = base_dir / Path('figures')
res_path = base_dir / Path('results')

exp_suffix = 'SIREN_32x32x32'
# train
batch_size = {
    'train' : 128
,   'val'   : 128
}
dx = {
    'train' : 8
,   'val'   : 4
}
dt = {
    'train' : 8 
,   'val'   : 4
}
nt_train = 4

run = 'SGU'

model = {
    'name' : 'Siren' # 'RFFMLP
,   'cnn' : False
}

n_epoch = 5000
seed = 666
hparams = {
    'name': model['name'],
    'do_time' : True,
    'do_cnn' : model['cnn'],
    'mlp_layers_num': 5,
    'mlp_layers_dim': 128, 
    'mlp_last_actfn': 'tanh',
    'rff_num_space': 128,
    'rff_scale_space': 10,
    'rff_num_time': 128, 
    'rff_scale_time': 10,
    'do_divfree': False,
    'lam_sdiv': 0, 
    'lam_sfn':  0,    
    'lam_spec': 0,    
    'lam_grads':0,    
    'lam_curl' :0,    
    'lam_pde' : 0,
    'lam_weight': 1e-5,  # L2 reg on the NN's weights
    'sfn_min_x': 1./256., # maximal resolution
    'sfn_num_centers': 32,
    'sfn_num_increments':8,
    'sfn_patch_dim': 16 # (P/2)
}

def main():

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
    

if __name__ == '__main__':
    
    model, best_model_path, hparams = main()
    print(best_model_path.split('/')[-1])

    results = {
        'exp_name' : exp_suffix
    ,   'best_model_path' : best_model_path
    ,   'hparams' : hparams
    }
    
    results_path = res_path / Path(f'results_{exp_suffix}.yaml')
    with open(results_path, 'w') as file:

        outputs = yaml.dump(results, file)

    print('Done.')
import gc
import yaml

from torchsummary import summary
from argparse import Namespace
from pathlib import Path
import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning.trainer import seed_everything

from turboflow.dataloaders import TurboFlowDataModule
from turboflow.models.phyrff import plDivFreeRFFNet

import matplotlib.pyplot as plt

base_dir = Path('/','home','dicarlo_d','Documents','Code','TurboSuperResultion','recipes','time_cnn')
data_dir = Path('/','home','dicarlo_d','Documents','Datasets','Turb2D.hdf5')
fig_path = base_dir / Path('figures')
res_path = base_dir / Path('results')

dx_space = {
    'train' : 8
,   'val'   : 4
,   'test'  : 2
}
nt_space = 4

dt_time = {
    'train' : 8 
,   'val'   : 4
,   'test'  : 2
}
nt_time = 32
seed = 666

def test_time_space(model,
                dx_train=8, 
                dx_val=8, 
                dx_test=8, 
                dt_train=8, 
                dt_val=8, 
                dt_test=8, 
                n_time=10,
                batch_size=32):

    # setup dataset for TEST
    # TEST spatial super-resolution
    dm_test_space = TurboFlowDataModule(dataset='Turb2D', 
                            data_dir=data_dir,
                            time_idx=np.arange(n_time * dt_train),
                            train_batch_size=int(batch_size),
                            val_batch_size=int(batch_size),
                            test_batch_size=int(batch_size),

                            train_downsampling_space=dx_train,
                            val_downsampling_space=dx_val,
                            test_downsampling_space=dx_test,

                            train_downsampling_time=dt_train,
                            val_downsampling_time=dt_val,
                            test_downsampling_time=dt_test,

                            train_shuffle=False,
                            val_shuffle=False,
                            test_shuffle=False,
                            num_workers=0)
    dm_test_space.setup()

    dataloaders = [
        {'name':'train','dl':dm_test_space.train_dataloader,'dset':dm_test_space.train_dataset},
        {'name':'val','dl':dm_test_space.val_dataloader,'dset':dm_test_space.val_dataset},
        {'name':'test','dl':dm_test_space.test_dataloader,'dset':dm_test_space.test_dataset}
    ]

    res = {}

    for dl in dataloaders:

        name = dl['name']
        print(name)
        uv_true = []
        uv_estm = []
        txy = []
        
        img_shape = dl['dset'].img_shape
        vshape = dl['dset'].vars_shape_img
        fshape = dl['dset'].fields_shape_img
        
        for X, y in dl['dl']():

            y_hat, P_hat = model(X.to(model.device))
            
            uv_true.append(y.detach().cpu().numpy())
            uv_estm.append(y_hat.detach().cpu().numpy())
            txy.append(X)
            
        txy = np.concatenate(txy, axis=0)    
        uv_true = np.concatenate(uv_true, axis=0)
        uv_estm = np.concatenate(uv_estm, axis=0)
        
        res[name] = {
            'uv_true' : uv_true.reshape(*fshape),
            'uv_estm' : uv_estm.reshape(*fshape),
        }

    del dm_test_space
    del dataloaders
    collected = gc.collect()
    print("Garbage collector: collected %d objects." % collected)


    return res

def test(results_yml):
    print(results_yml)
    with open(results_yml, 'r') as file:
        result = yaml.safe_load(file)

    best_model_path = result['best_model_path']
    exp_name = result['exp_name']

    seed_everything(seed, workers=True)
    model = plDivFreeRFFNet.load_from_checkpoint(best_model_path)
    # summary(model, (1, 3), batch_size=16, device='cpu')
    model.cuda()

    print('TEST SPACE')
    res_space = test_time_space(
            model
        ,   dx_train=dx_space['train']
        ,   dx_val=dx_space['val']
        ,   dx_test=dx_space['test']
        ,   n_time=nt_space
        ,   batch_size=128
    )

    fig, axarr = plt.subplots(3, 3, figsize=(10,10))

    for s, stage in enumerate(res_space.keys()):
        
        uv_true = res_space[stage]['uv_true']
        uv_estm = res_space[stage]['uv_estm']

        t = 0
        axarr[s,0].set_title('VEL groundtruth')
        axarr[s,0].imshow(uv_true[t,:,:,0])
        
        axarr[s,1].set_title('VEL predicted')
        axarr[s,1].imshow(uv_estm[t,:,:,0])

        axarr[s,2].set_title('Y view')
        axarr[s,2].plot(uv_true[t,:,0,0], label='true')
        axarr[s,2].plot(uv_estm[t,:,0,0], label='estm')

    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path / Path('{}_{}_space.png'.format(model.name, exp_name)))
    plt.show()

    print('TEST TIME')
    res_time  = test_time_space(
                model
            ,   dt_train=dt_time['train']
            ,   dt_val=dt_time['val']
            ,   dt_test=dt_time['test']
            ,   n_time=nt_time
            ,   batch_size=128)

    fig, axarr = plt.subplots(3,1, figsize=(5,5))
    for s, stage in enumerate(res_time.keys()):
        
        uv_true = res_time[stage]['uv_true']
        uv_estm = res_time[stage]['uv_estm']

        i = 16
        j = 8
        
        axarr[s].plot(uv_true[:,i,j,0], label='VEL groundtruth')
        axarr[s].plot(uv_estm[:,i,j,0], label='VEL predicted')

    plt.tight_layout()
    plt.savefig(fig_path / Path('{}_{}_time.png'.format(model.name, exp_name)))
    plt.show()



if __name__ == '__main__':

    results_file = res_path.glob('*.yaml')

    print('Which file?')
    results_file = sorted(list(results_file))
    for f, file in enumerate(results_file):
        name = file.name
        print(f'[{f}]\t{name}')

    idx = int(input())
    test(results_file[idx])
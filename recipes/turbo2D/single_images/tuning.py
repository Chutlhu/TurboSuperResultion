from argparse import Namespace

import pickle
from pathlib import Path

from pytorch_lightning.trainer import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from turboflow.dataloaders import TurboFlowDataModule
from turboflow.evaluation import compute_all_metrics

from turboflow.utils import phy_utils as phy
from turboflow.utils import torch_utils as tch
from turboflow.utils import file_utils as fle

################################# CONFIG ######################################
data = 'Turb2D'
hparams = {
    'name':'RFF:4096-softDF-Sfun:1e-1',
    'mlp_layers_num': 3, 
    'mlp_layers_dim': 256, 
    'mlp_last_actfn': 'tanh',
    'do_rff': True, 
    'rff_num': 4096, 
    'rff_scale': 10,
    'do_divfree': False,
    'lam_pde': 1e-4, 
    'lam_div': 0,
    'lam_reg': 0,
    'lam_sfn': 1e-1,
    'lam_spec': 0,
    'sfn_min_x': 0.00784314,
    'sfn_num_centers': 100,
    'sfn_num_increments':4,
    'sfn_patch_dim': 50
}
exp_name = '%s-%s' % (data, hparams['name'])
results_dir = Path('.','results',f'{exp_name}.pkl')

max_epochs = 5000
check_val_every_n_epoch = 50
############################## END CONFIG #####################################

seed_everything(42, workers=True)

data_dir = '/home/dicarlo_d/Documents/Code/TurboSuperResultion/.cache/Turb2D.hdf5'
dm = TurboFlowDataModule(dataset='Turb2D', 
                         data_dir=data_dir,
                         batch_size=100000,
                         time_idx=33,
                         train_downsampling=4,
                         val_downsampling=4,
                         test_downsampling=1,
                         num_workers=1)
dm.setup()

early_stop_callback = EarlyStopping(monitor='val_loss')
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=".torch_checkpoints",
    filename="%s-%s-{epoch:02d}-{val_loss:.2f}" % (data, hparams['name']),
    save_top_k=2,
    mode="min",
)

trainer = Trainer(gpus=1,
                  max_epochs=max_epochs, 
                  log_every_n_steps=1,
                  check_val_every_n_epoch=check_val_every_n_epoch, 
                  callbacks=[early_stop_callback,
                             checkpoint_callback])

path_to_tb = Path(trainer.logger.save_dir, 'version_%s' % (trainer.logger.version))

from turboflow.models.phyrff_hard import plDivFreeRFFNet
model = plDivFreeRFFNet(**vars(Namespace(**hparams)))

print('Running', hparams['name'], 'aka', 'version_%s' % (trainer.logger.version))

trainer.fit(model, dm)
results = trainer.test(model, dm)

## Save Results
exp_dict = {
    'path_to_model' : checkpoint_callback.best_model_path,
    'model' : model.hparams,
    'name' : exp_name,
    'results' : results,
    'tb_trial_id' : str(path_to_tb),
}


fle.save_obj(exp_dict, results_dir)
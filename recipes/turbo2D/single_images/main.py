import argparse

import os
from pathlib import Path

# DNN Libraries
# import pl torch libs
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# Turboflow Libraries
from turboflow.dataloaders import TurboFlowDataModule
from turboflow.models.phyrff_hard import plDivFreeRFFNet

from turboflow.utils.file_utils import save_obj


def get_path_and_prepare_folder():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    return root_dir

def get_args():

    # these are project-wide arguments
    parser = argparse.ArgumentParser(prog='main.py', add_help=True)

    group = parser.add_argument_group("exp")
    group.add_argument("--exp_name", type=str, required=True)
    group.add_argument("--logs_dir", type=str, required=True)
    group.add_argument("--res_dir", type=str, required=True)
    
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
    exp_params_dict = vars(hparams['exp'])
    res_dir = exp_params_dict['res_dir']
    exp_name = exp_params_dict['exp_name']
    log_dir = exp_params_dict['logs_dir']

    # PREPARE FOLDER
    root_dir = Path(get_path_and_prepare_folder())

    dm = TurboFlowDataModule.from_argparse_args(hparams['data'])

    # MODEL
    tb_logger = pl_loggers.TensorBoardLogger(log_dir)
    model_param_dict = vars(hparams['model'])
    model = plDivFreeRFFNet(**model_param_dict)
    early_stop_callback = EarlyStopping(monitor='val/loss/tot', min_delta=1e-5)
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss/tot",
        dirpath=".torch_checkpoints",
        filename="Turb2D-%s-{epoch:02d}-{val_loss:.2f}" % (model_param_dict['name']),
        save_top_k=1,
        mode="min",
    )
    trainer = Trainer.from_argparse_args(
                hparams['pl.Trainer'],
                logger=tb_logger,
                callbacks=[early_stop_callback, checkpoint_callback])

    # TRAIN
    trainer.fit(model, dm)
    
    # TEST
    test_metrics = trainer.test()
    best_model_path = checkpoint_callback.best_model_path
    success = True
    try:
        pass
    except Exception as e:
        print(e)
        test_metrics = {}
        success = False
        best_model_path = ''

    res_dict = {
        'path_to_best_model' : best_model_path,
        'test_metrics' : test_metrics,
        'hparams' : model_param_dict,
        'pl_version' : f'version_{trainer.logger.version}',
        'success' : success,
    }
    out_file = '%s/Turb2D_%s.pkl' % (res_dir, exp_name)
    out_file = root_dir / Path(out_file)
    save_obj(res_dict, out_file)
    print(f'Results in: {out_file}')
    print('Done!')
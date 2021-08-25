import torch
from pytorch_lightning.callbacks import Callback


class RefineSolution(Callback):
    def __init__(self, thr: float = 1e-5):
        super().__init__()
        self.monitor = 'valid_loss'
        self.thr = thr
        self.is_enabled = False
    
    def on_validation_end(self, trainer, model):
        logs = trainer.callback_metrics
        if (not self.is_enabled) and logs.get(self.monitor) < 1e-6:
            print('Switched to LBFGS')
            trainer.optimizers = [torch.optim.LBFGS(model.parameters(), lr=1e-4)]
            self.is_enabled = True
            # trainer.lr_schedulers = trainer.configure_schedulers([new_schedulers])
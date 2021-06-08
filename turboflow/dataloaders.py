import torch
import numpy as np
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader

class FFMDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, input_img, mapping=None):
        'Initialization'
        
        # on single image
        assert input_img.shape[0] == input_img.shape[1]
        img_size = input_img.shape[0]
        
        coords = np.linspace(0, 1, input_img.shape[0], endpoint=False)
        coords = np.stack(np.meshgrid(coords, coords), -1) # X x Y x 2
        self.inputs = torch.from_numpy(coords).float() #XY x 2 (x,y)
        
        if len(input_img.shape) == 2:
            input_img = input_img[:,:,None]
            
        self.target = torch.from_numpy(input_img).float()  #XxYx3 (RGB)
        
        assert self.inputs.shape[0] == self.target.shape[0]
        assert self.inputs.shape[1] == self.target.shape[1]

            
  def __len__(self):
        'Denotes the total number of samples'
        return 1

    
  def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = self.inputs
        y = self.target

        return X, y
    

class FFMDataModule(pl.LightningDataModule):
    def __init__(self, img, training_downsampling_factor=2, batch_size = 1):
        
        super().__init__()
        self.batch_size = 1
        self.training_downsampling_factor = training_downsampling_factor
        self.img = img
        try:
            assert np.max(np.abs(img)) <= 1.0
        except:
            raise ValueError('Image is not within [-1, 1]. Max is %1.2f' % np.max(np.abs(img)))
        
    def prepare_data(self):
        self.train_dataset = FFMDataset(np.array(self.img[::self.training_downsampling_factor,::self.training_downsampling_factor]))
        self.valid_dataset = FFMDataset(np.array(self.img[::self.training_downsampling_factor,::self.training_downsampling_factor]))
        self.test_dataset = FFMDataset(np.array(self.img))
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

if __name__ == '__main__':
    pass
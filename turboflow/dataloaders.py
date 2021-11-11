import glob
import torch
import numpy as np

from natsort import natsorted
from torch.utils import data

from turboflow.datasets.turb2D import Turb2D

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

currently_supported_dataset = ['Turb2D', 'Re39000']

class Re39000Dataset(Dataset):

    def __init__(self, data_dir:str, ds:int, time_idx:int, z=1):
        super(Re39000Dataset, self).__init__()
        
        self.name = "Re39000"

        self.time_idx = time_idx
        # load raw data
        files = natsorted(list(glob.glob(data_dir + "*.txt")))
        file = files[time_idx]
        data = np.loadtxt(file)

        self.size = data.shape[0]

        # hardcoded data processing
        self.n_vars = 6 # x, y, z, u, v, w
        self.nx = 308
        self.ny = 328
        self.nz = 3
        data = data.reshape(self.nz, self.nx, self.ny, self.n_vars)
        data = data.traspose(1,2,0,3)

        # hardcoded data downsampling
        #             x     y   z  vars
        data = data[::ds, ::ds, z, :][:,:,None,:]
        self.nx = data.shape[0]
        self.ny = data.shape[1]
        self.nz = data.shape[2]

        self.order_x = "xyz"
        self.order_y = "uvw"

        self.X = torch.from_numpy(data[...,:3]).float().view(-1, 2) # x, y, z
        self.y = torch.from_numpy(data[...,3:]).float().view(-1, 2) # u, v, w

        assert self.X.shape == self.y.shape

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        X = self.X[idx,:]
        y = self.y[idx,:]
        return X, y


class Turb2DDataset(Dataset):
    def __init__(self,data_dir:str=None,ds:int=1,dt:int=1,time_idx:int=None):

        tb = Turb2D(data_dir)
        tb.setup()
        tb.load_data(time_idx)

        # Data in Turb2D are (T,R,R,D)
        t = tb.t
        X = tb.xy
        y = tb.uv

        # normalize y
        y = y/np.max(np.abs(y))
        assert np.min(y) >= -1
        assert np.max(y) <=  1

        assert X.shape[0] == y.shape[0]
        assert len(X.shape) in [3,4]

        if len(X.shape) == 3: # single image/time
            
            # downsampling
            X = X[::ds, ::ds, :]
            y = y[::ds, ::ds, :]
        
            self.res = X.shape[0] 
            self.img_shape = X.shape # (R,R,2)

            self.X = torch.from_numpy(X).float().view(-1,2)
            self.y = torch.from_numpy(y).float().view(-1,2)
            self.t = torch.from_numpy(t).float()
            self.size = self.X.shape[0]

        if len(X.shape) == 4: # multiple images/times
            # downsampling
            X = X[::dt, ::ds, ::ds, :]
            y = y[::dt, ::ds, ::ds, :]
            t = t[::dt]
        
            self.times = X.shape[0]
            self.res = X.shape[1] 
            self.img_shape = X.shape[1:3] # (R,R,2)

            self.X = torch.from_numpy(X).float().view(-1,2)
            self.t = torch.from_numpy(t).float()
            self.y = torch.from_numpy(y).float().view(-1,2)
            self.size = self.X.shape[0]

        assert self.X.shape == self.y.shape
                
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        X = self.X[idx,:]
        y = self.y[idx,:]
        return X, y


class TurboFlowDataModule(pl.LightningDataModule):
    def __init__(self, dataset:str, data_dir:str, batch_size:int, time_idx:int,
                 train_downsampling:int, val_downsampling:int, test_downsampling:int,
                 num_workers:int):
        super(TurboFlowDataModule, self).__init__()

        self.dataset = dataset

        if not self.dataset in currently_supported_dataset:
            raise ValueError(f'Supported Dataset are {currently_supported_dataset}, got: {dataset}')

        if self.dataset == 'Turb2D':
            self.dataset_fn = Turb2DDataset
        if self.dataset == 'Re39000':
            self.dataset_fn = Re39000Dataset

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.time_idx = time_idx
        self.train_ds = train_downsampling
        self.val_ds = val_downsampling
        self.test_ds = test_downsampling
        self.num_workers = num_workers

    @staticmethod
    def add_data_specific_args(parent_parser):
        group = parent_parser.add_argument_group("data")
        group.add_argument("--dataset", type=str, required=True)
        group.add_argument("--data_dir", type=str, required=True)
        group.add_argument("--train_downsampling", type=int, default=4)
        group.add_argument("--val_downsampling", type=int, default=4)
        group.add_argument("--test_downsampling", type=int, default=1)
        group.add_argument("--time_idx", type=int, default=42)
        group.add_argument("--batch_size", type=int, default=100000)
        group.add_argument("--num_workers", type=int, default=1)
        return parent_parser
        
    def prepare_data(self):
        # if download is required
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset_fn(self.data_dir, self.train_ds, self.time_idx)
            self.val_dataset = self.train_dataset

        if stage == "test" or stage is None:
            self.test_dataset = self.dataset_fn(self.data_dir, self.test_ds, self.time_idx)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, num_workers=self.num_workers)


def load_turb2D_simple_numpy(path_to_turb2D:str='../data/2021-Turb2D_velocities.npy',ds:int=4,img:int=42):

    IMGs = np.load(path_to_turb2D)
    X = IMGs[img,::ds,::ds,:2] / 255
    y = IMGs[img,::ds,::ds,2:]

    # normalize output
    print('Y shape', y.shape)
    print('Y min, max:', np.min(y), np.max(y))
    y = y / np.max(np.abs(y))
    print('after normalization, Y min, max:', np.min(y), np.max(y))

    X = X.reshape(-1,2)
    y = y.reshape(-1,2)

    assert X.shape == y.shape

    return X, y


# class Turb2D_simple(Dataset):
    
#     def __init__(self, path_to_turb2D, device, ds=4, img=42):
        
#         print('Dataset Turb2D, img #', img)

#         IMGs = np.load(path_to_turb2D)
#         X = IMGs[img,::ds,::ds,:2] / 255
#         U = IMGs[img,::ds,::ds,2:]

#         print(X.shape)
#         print(U.shape)

#         original_size = X.shape[0]
#         print('Original size', original_size)

#         # normalize output
#         y = U.copy()
#         print('Y shape', y.shape)
#         print('Y min, max:', np.min(y), np.max(y))
#         y = y / np.max(np.abs(y))
        
#         print('after normalization, Y min, max:', np.min(y), np.max(y))

#         self.x = torch.from_numpy(X).float().to(device).view(-1,2)
#         self.y = torch.from_numpy(y).float().to(device).view(-1,2)

#         assert self.x.shape[0] == self.y.shape[0]

    
#     def __len__(self):
#         return self.x.shape[0]
    

#     def __getitem__(self, idx):
#         x = self.x[idx,:]
#         y = self.y[idx,:]
#         return (x, y)


if __name__ == '__main__':
    pass




# class Turb2D_simple_with_neighbours(Dataset):
#     def __init__(self, path_to_turb2D, device, ds=4, img=42):
        
#         print('Dataset Turb2D, img #', img)

#         IMGs = np.load(path_to_turb2D)
#         X = IMGs[img,::ds,::ds,:2] / 255
#         U = IMGs[img,::ds,::ds,2:]

#         print(X.shape)
#         print(U.shape)

#         original_size = X.shape[0]
#         print('Original size', original_size)

#         # normalize output
#         y = U.copy()
#         print('Y shape', y.shape)
#         print('Y min, max:', np.min(y), np.max(y))
#         y = y / np.max(np.abs(y))
        
#         print('after normalization, Y min, max:', np.min(y), np.max(y))

#         self.x = torch.from_numpy(X).float().to(device).view(-1,2)
#         self.y = torch.from_numpy(y).float().to(device).view(-1,2)

#         # make context patches of 3x3
#         L = X.shape[0]
#         P = 3
#         assert P == 3
#         self.c = torch.zeros((L,L,P,P))
#         for i in range(L):
#             for j in range(L):
#                 for ii in range(P):
#                     for jj in range(P):
#                         self.c[i,j,ii,jj] = self.y[ii-1,jj-1]
#         self.c = self.c.view(-1,P*P)


#         assert self.x.shape[0] == self.y.shape[0]

    
#     def __len__(self):
#         return self.x.shape[0]
    

#     def __getitem__(self, idx):
#         x = self.x[idx,:]
#         c = self.c[idx,:]
#         y = self.y[idx,:]
#         return (x, c, y)




# class FFMDataset(torch.utils.data.Dataset):
#   'Characterizes a dataset for PyTorch'
#   def __init__(self, input_img, mapping=None):
#         'Initialization'
        
#         # on single image
#         assert input_img.shape[0] == input_img.shape[1]
#         img_size = input_img.shape[0]
        
#         eps = 1e-15

#         coords = np.linspace(eps, 1, input_img.shape[0], endpoint=False)
#         coords = np.stack(np.meshgrid(coords, coords), -1) # X x Y x 2
#         self.inputs = torch.from_numpy(coords).float() #XY x 2 (x,y)
        
#         if len(input_img.shape) == 2:
#             input_img = input_img[:,:,None]
            
#         self.target = torch.from_numpy(input_img).float()  #XxYx3 (RGB)
        
#         assert self.inputs.shape[0] == self.target.shape[0]
#         assert self.inputs.shape[1] == self.target.shape[1]

            
#   def __len__(self):
#         'Denotes the total number of samples'
#         return 1

    
#   def __getitem__(self, index):
#         'Generates one sample of data'
#         # Load data and get label
#         X = self.inputs
#         y = self.target

#         return X, y
    
    
# class FFMDataModule(pl.LightningDataModule):
#     def __init__(self, img, training_downsampling_factor=2, batch_size = 1):
        
#         super().__init__()
#         self.batch_size = 1
#         self.training_downsampling_factor = training_downsampling_factor
#         self.img = img
#         try:
#             assert np.max(np.abs(img)) <= 1.0
#         except:
#             raise ValueError('Image is not within [-1, 1]. Max is %1.2f' % np.max(np.abs(img)))
        
#     def prepare_data(self):
#         self.train_dataset = FFMDataset(np.array(self.img[::self.training_downsampling_factor,::self.training_downsampling_factor]))
#         self.valid_dataset = FFMDataset(np.array(self.img[::self.training_downsampling_factor,::self.training_downsampling_factor]))
#         self.test_dataset = FFMDataset(np.array(self.img))
    
#     def train_dataloader(self):
#         return DataLoader(self.train_dataset, batch_size=self.batch_size)

#     def val_dataloader(self):
#         return DataLoader(self.valid_dataset, batch_size=self.batch_size)

#     def test_dataloader(self):
#         return DataLoader(self.test_dataset, batch_size=self.batch_size)


# class MyDataset(torch.utils.data.Dataset):
#   'Characterizes a dataset for PyTorch'
#   def __init__(self, X, y):

#         assert np.max(np.abs(y)) <= 1
#         # assert np.max(np.abs(X)) <= 1

#         'Initialization'
#         print(X.shape)
#         print(y.shape)
#         try:
#             assert X.shape[0] == y.shape[0]
#             assert X.shape[1] == y.shape[1]
#         except:
#             print('Dimension error')
#             print('X', X.shape)
#             print('y', y.shape)

#         # convert to torch and add an empty dimension for the batch
#         self.inputs = torch.from_numpy(X).float() #XY x 2 (x,y)
#         self.target = torch.from_numpy(y).float()  #XxYx3 (RGB)
        
#         assert self.inputs.shape[0] == self.target.shape[0]
#         assert self.inputs.shape[1] == self.target.shape[1]

            
#   def __len__(self):
#         'Denotes the total number of samples'
#         return 1

    
#   def __getitem__(self, index):
#         'Generates one sample of data'
#         # Load data and get label
#         X = self.inputs
#         y = self.target

#         return X, y



# class MyPatchDataset(torch.utils.data.Dataset):
#   'Characterizes a dataset for PyTorch'
#   def __init__(self, X, y):

#         'Initialization'
#         self.X = X
#         self.y = y

#         assert len(X) == len(y)

#         self.size = len(X)


#   def __len__(self):
#         'Denotes the total number of samples'
#         return self.size

    
#   def __getitem__(self, index):
#         'Generates one sample of data'
#         # Load data and get label
#         X = torch.from_numpy(self.X[index]).float() #XY x 2 (x,y)
#         y = torch.from_numpy(self.y[index]).float()  #XxYx3 (RGB)

#         return X, y





# class PatchDataModule(pl.LightningDataModule):
#     def __init__(self, train_data, val_data, test_data, batch_size=64):
        
#         super().__init__()
#         self.batch_size = 1
#         self.train_data = train_data
#         self.test_data = test_data
#         self.val_data = val_data
#         self.batch_size = batch_size
        
#     def prepare_data(self):
#         self.train_dataset = MyPatchDataset(self.train_data[0], self.train_data[1])
#         self.val_dataset = MyPatchDataset(self.val_data[0], self.val_data[1])
#         self.test_dataset = MyPatchDataset(self.test_data[0], self.test_data[1])
    
#     def train_dataloader(self):
#         return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

#     def val_dataloader(self):
#         return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)

#     def test_dataloader(self):
#         return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)
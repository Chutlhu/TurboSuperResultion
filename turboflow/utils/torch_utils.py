import torch

def get_device(verbose=True):
    torch.cuda.is_available()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if verbose: print('Torch running on:', device)
    return device

def to_numpy(x):
    return x.detach().cpu().numpy()

def to_torch(x, device):
    return torch.from_numpy(x).float().to(device)
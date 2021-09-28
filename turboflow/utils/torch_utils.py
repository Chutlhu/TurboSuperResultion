import torch

def get_device():
    torch.cuda.is_available()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print('Torch running on:', device)
    return device
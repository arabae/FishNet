import yaml
import torch
import pickle
import random
import numpy as np


class ColorAugmentation(object):
    def __init__(self):
        self.eig_vec = torch.Tensor([
            [-0.183, -0.2333, -0.5551],
            [0.0007, 0.0123, -0.5534],
            [0.1902, 0.2367, -0.5317],
        ])
        self.eig_val = torch.Tensor([[0.0006, 0.0058, 0.1267]])

    def __call__(self, tensor):
        assert tensor.size(0) == 3
        alpha = torch.normal(mean=torch.zeros_like(self.eig_val)) * 0.1
        quatity = torch.mm(self.eig_val * alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor

def load_data(filepath, whole_dataset, num_data_batch):
    read_data = unpickle(filepath)

    for k, v in read_data.items():
        k = k.decode('utf-8')
        
        if k == 'batch_label':
            v = [v.decode('utf-8')] * num_data_batch
        
        elif k == 'filenames':
            v = list(map(lambda x: x.decode('utf-8'), v))
        
        whole_dataset[k].extend(v)

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def unpickle(filepath):
    with open(filepath, 'rb') as fp:
        load_dict = pickle.load(fp, encoding='bytes')
    return load_dict

import torch
import torch.utils.data as data

class Dataset(data.Dataset):
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels
        self.trainmode = True if labels is not None else False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.trainmode:
            return torch.Tensor(self.data[index]),torch.Tensor(self.labels[index])
        else:
            return torch.Tensor(self.data[index])

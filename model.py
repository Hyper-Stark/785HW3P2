import torch
import torch.nn as nn
import torch.nn.functional as F
from data.phoneme_list import *
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_packed_sequence

class Model(nn.Module):

    def __init__(self, device = None, input_size = 40, output_size = len(PHONEME_MAP) + 1, num_layers=4, hidden_size = 512):
        super(Model,self).__init__()
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional = True)
        self.classifier = nn.Sequential(
            nn.Linear(2*hidden_size,hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )
        self.to(device)

    def forward(self, x):
        packedres = pack_sequence(x)
        packedres = packedres.to(self.device)
        lstmres, (hidden,cell) = self.lstm(packedres, None)
        unpackedres, __ = pad_packed_sequence(lstmres)
        output = self.classifier(unpackedres)
        return output
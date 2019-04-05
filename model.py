import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from IPython.core.debugger import set_trace

OUT_DIM = 47

class Model(nn.Module):

    def __init__(self,device,layers=4,bidir=True):
        super(Model,self).__init__()
        
        self.device = device
        self.num_layers = layers
        self.input_size = 256
        self.mid_size = 128
        self.hidden_size = 256
        self.direction = 1
        if bidir:
            self.direction = 2

        
        self.conv1 = nn.Conv1d(40,self.mid_size,kernel_size =32,stride=2,bias=False)
        self.conv2 = nn.Conv1d(self.mid_size,self.input_size,kernel_size =16,stride=1,bias=True)
        self.batch1 = nn.BatchNorm1d(self.mid_size)
        

        self.lstm = nn.LSTM(
            input_size = self.input_size,
            hidden_size = self.hidden_size,
            dropout=0.2,
            num_layers=layers, 
            bidirectional=bidir,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.input_size*self.direction, self.mid_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.mid_size, OUT_DIM)
        )

    # def initHiddenCell(self,batch_size):

    #     dim = (self.direction * self.num_layers, batch_size, self.hidden_size)
    #     rand_hidden = Variable(torch.randn(*dim)).to(self.device)
    #     rand_cell = Variable(torch.randn(*dim)).to(self.device)
    #     return rand_hidden, rand_cell

    def forward(self, x, lengths):
        #'''
        conv1 = self.conv1(x)
        batch1 = self.batch1(conv1)
        convres1 = F.relu(batch1)
        convres2 = F.relu(self.conv2(convres1))
        convres2 = convres2.permute(0,2,1)
        #'''
        #convres2 = x.permute(0,2,1)
        lengths = [convlen(convlen(l,self.conv1),self.conv2) for l in lengths]
        packedres = pack_padded_sequence(convres2,lengths,batch_first=True)
        lstmres, (hidden, cell) = self.lstm(packedres)
        unpackedres, __ = pad_packed_sequence(lstmres,batch_first=True)

        output = self.classifier(unpackedres)
        return output, lengths

def convlen(length,conv):
    return (length - conv.kernel_size[0])//conv.stride[0] + 1

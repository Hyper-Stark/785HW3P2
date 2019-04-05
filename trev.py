import pytz
import time
import math
import torch
import torchvision
import numpy as np
import torch.nn as nn
import datetime as dt
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

from model import Model
from dataset import Dataset
from utils import details
from utils import pad_at_end
from ctc import Predictor 

from data.phoneme_list import *
from ctcdecode import CTCBeamDecoder
from data.phoneme_list import PHONEME_MAP
 
EPOCHS = 12
BATCH_SIZE = 128
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tz = pytz.timezone("America/New_York")
_begin_time = time.time()

#prepare data
# train
traindata = np.load("data/wsj0_train.npy",encoding="bytes")
trainlabel = np.load("data/wsj0_train_merged_labels.npy",encoding="bytes")
#traindata = np.load("data/wsj0_dev.npy",encoding="bytes")
#trainlabel = np.load("data/wsj0_dev_merged_labels.npy",encoding="bytes")
trainset = Dataset(traindata,trainlabel)
# valid
validdata = np.load("data/wsj0_dev.npy",encoding="bytes")
validlabel = np.load("data/wsj0_dev_merged_labels.npy",encoding="bytes")
validset = Dataset(validdata,validlabel)

# test
testdata = np.load("data/transformed_test_data.npy",encoding="bytes")
testset = Dataset(testdata)

# loaders
train_loader = data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=pad_at_end, num_workers=4)
valid_loader = data.DataLoader(validset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, collate_fn=pad_at_end, num_workers=4)
test_loader = data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, collate_fn=pad_at_end, num_workers=4)

_end_time = time.time()
print("load data time cost: " + str(_end_time - _begin_time))
_begin_time = time.time()

# prepare model
model = Model(DEVICE).to(DEVICE)
criterion = nn.CTCLoss()
opt = optim.Adam(model.parameters(),lr=0.001,weight_decay=5e-4)
predictor = Predictor()

_end_time = time.time()
print("prepare model: " + str(_end_time - _begin_time))
_begin_time = time.time()

#train
for epoch in range(EPOCHS):

    model.train()

    #this epoch
    nt = dt.datetime.now(tz)
    print(" ")
    print("Starting epoch "+str(epoch)+" at "+nt.strftime("%H:%M"))
    # print("current learning rate: "+str(scheduler.get_lr())) 

    # statistic variables
    i = ave_loss = avg_dis = total = 0
    for voice,labels,lengths in train_loader:
       
        #clear gradients
        print("    training batch: "+str(i))
        opt.zero_grad()
        #forward
        output, lengths = model(voice.to(DEVICE),lengths)
        output = output.permute(1,0,2)
        log_probs = output.log_softmax(2)

        print("    training forward")
        #concate labels
        len_of_labels = []
        concat_labels = torch.tensor([],dtype=torch.int)
        for l in labels:
            len_of_labels.append(l.shape[0])
            concat_labels = torch.cat((concat_labels,l),0)

        #loss
        labels = concat_labels.to(DEVICE)
        loss = criterion(log_probs, labels, tuple(lengths), tuple(len_of_labels))
        #accumulate loss
        nl = loss.item()
        print("    new loss: "+str(nl))
        if not math.isnan(nl):
            ave_loss += nl
        #accumulate L distance
        
        print("    training backward")
        #backward
        loss.backward()
        #gradient decent update
        opt.step()

        total += output.shape[0]
        i += 1
    
    #eval mode for valid data set
    model.eval()

    #statistic variables for validation
    vtotal = vavdis = vi = vloss = 0
    for voice, labels, lengths in valid_loader:
 

        print("    eval forward") 
        #evaluate
        output, lengths = model(voice.to(DEVICE),lengths)
        output = output.permute(1,0,2)
        log_probs = output.log_softmax(2)

        #concate labels
        len_of_labels = []
        concat_labels = torch.tensor([],dtype=torch.int)
        for l in labels:
            len_of_labels.append(l.shape[0])
            concat_labels = torch.cat((concat_labels,l),0)

        #loss
        labels = concat_labels.to(DEVICE)
        loss = criterion(log_probs, labels, tuple(lengths), tuple(len_of_labels))

        print("    eval calculate distance") 
        #count accuracy
        nl = loss.item()
        print("    new loss: "+str(nl))
        if not math.isnan(nl):
            vloss += nl
        
        vavdis += predictor.evaluateError(output, labels, len_of_labels)
        vtotal += output.shape[0]
        vi += 1

        '''

        print("test")

    '''
    _end_time = time.time()
    details((_end_time - _begin_time), ave_loss/i, avg_dis/total, vloss/vi, vavdis/vtotal)
    _begin_time = time.time()

    torch.save(model.state_dict(),"models/model-"+str(int(_begin_time))+str(int(vavdis/vtotal))+".pt")

model.eval()
result = []
for voice,lengths in test_loader:
    #evaluate
    output, lengths = model(voice.to(DEVICE),lengths)
    preds = predictor.predict(output)
    result = result + preds 

_end_time = time.time()
print("evaluation cost time: "+str(_end_time - _begin_time))
_begin_time = time.time()  

#write to file
with open("result.csv",'w') as f:
    f.write("Id,Predicted\n")
    for i,item in enumerate(result):
        f.write(','.join([str(i),item]) + '\n')


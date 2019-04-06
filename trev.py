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
from utils import loader,details
from ctc import Predictor 

#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')
 
EPOCHS = 12
TIME_ZONE = pytz.timezone("America/New_York")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load data
train_loader = loader("data/wsj0_train.npy","data/wsj0_train_merged_labels.npy")
#train_loader = loader("data/wsj0_dev.npy","data/wsj0_dev_merged_labels.npy")
valid_loader = loader("data/wsj0_dev.npy","data/wsj0_dev_merged_labels.npy")

_begin_time = time.time()
# prepare model
model = Model(device = DEVICE)
model.load_state_dict(torch.load("model.pt"))

criterion = nn.CTCLoss()
opt = optim.Adam(model.parameters(),lr=0.0001)
predictor = Predictor()

_end_time = time.time()
print("prepare model: " + str(_end_time - _begin_time))
_begin_time = time.time()

#train
for epoch in range(EPOCHS):

    model.train()

    #this epoch
    nt = dt.datetime.now(TIME_ZONE)
    print(" ")
    print("Starting epoch "+str(epoch)+" at "+nt.strftime("%H:%M"))
    # print("current learning rate: "+str(scheduler.get_lr())) 

    # statistic variables
    i = ave_loss = avg_dis = total = 0
    for voice,labels,lengths in train_loader:
       
        #clear gradients
        opt.zero_grad()
        #forward
        output = model(voice)
        log_probs = output.log_softmax(2)

        #concate labels
        len_of_labels = []
        concat_labels = torch.tensor([],dtype=torch.int)
        for l in labels:
            len_of_labels.append(l.shape[0])
            concat_labels = torch.cat((concat_labels,l),0)

        tlengths = torch.tensor(lengths).type(torch.IntTensor)
        tlen_of_labels = torch.tensor(len_of_labels).type(torch.IntTensor)
        #loss
        labels = concat_labels.to(DEVICE)
        loss = criterion(log_probs, labels, tlengths, tlen_of_labels)

        #accumulate loss
        nl = loss.item()
        if i % 16 == 0:
            print("    ["+str(epoch)+","+str(i)+"] loss: "+str(nl))
        if not math.isnan(nl):
            ave_loss += nl
        
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

        # forward
        output = model(voice)
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

        #count accuracy
        nl = loss.item()
        print("    ["+str(epoch)+","+str(vi)+"] loss: "+str(nl))
        if not math.isnan(nl):
            vloss += nl
        
        #vavdis += predictor.evaluateError(output, labels, len_of_labels)
        vtotal += output.shape[0]
        vi += 1

    _end_time = time.time()
    details((_end_time - _begin_time), ave_loss/i, avg_dis/total, vloss/vi, vavdis/vtotal)
    _begin_time = time.time()

    torch.save(model.state_dict(),"models/model-"+str(int(_begin_time))+"-"+str(int(vavdis/vtotal))+".pt")


result = []
model.eval()
test_loader = loader("data/transformed_test_data.npy")


for voice,lengths in test_loader:
    #evaluate
    output = model(voice)
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

import time
import torch
import numpy as np
import torch.utils.data as data

from dataset import Dataset
from torch.nn.utils.rnn import pad_sequence
from IPython.core.debugger import set_trace


BATCH_SIZE = 32
NUM_WORKERS = 4

def loader(dataf,labelf=None):

    _begin_time = time.time()
    dset = ld = None
    chunk = np.load(dataf,encoding="bytes")
    
    #train mode
    if labelf is not None:
        labels = np.load(labelf, encoding="bytes")
        dset = Dataset(chunk,labels)
        
        if torch.cuda.is_available():
            ld = data.DataLoader(\
                dset, \
                batch_size=BATCH_SIZE, \
                shuffle=True, \
                drop_last=True, \
                collate_fn=collate_train, \
                num_workers=NUM_WORKERS)
        else:
            ld = data.DataLoader(\
                dset, \
                batch_size=BATCH_SIZE, \
                shuffle=True, \
                drop_last=True, \
                collate_fn=collate_train)
    #test mode
    else:
        dset = Dataset(chunk)
        if torch.cuda.is_available():
            ld = data.DataLoader(\
                dset, \
                batch_size=1, \
                shuffle=False, \
                drop_last=False, \
                collate_fn=collate_test, \
                num_workers=NUM_WORKERS)
        else:
            ld = data.DataLoader(\
                dset, \
                batch_size=1, \
                shuffle=False, \
                drop_last=False, \
                collate_fn=collate_test)
    
    _end_time = time.time()
    print("load data time cost: " + str(_end_time - _begin_time))
    return ld

def collate_train(pairs):
 
    #split column
    inputs, labels = zip(*pairs)

    #collect lengths data
    seqlens = [(seq.shape[0], i) for i,seq in enumerate(inputs)]
    #sort indices according descending lengths
    sorted_seqlens = sorted(seqlens, key=lambda x: x[0], reverse=True)
    seqs,labs,lens = [],[],[]
    for lenz, oidx in sorted_seqlens:
        lens.append(lenz)
        seqs.append(inputs[oidx])
        labs.append(labels[oidx].type(torch.IntTensor) + 1)

    return seqs,labs,lens

def collate_test(inputs):

    #collect lengths data
    seqlens = [(seq.shape[0], i) for i,seq in enumerate(inputs)]
    #sort indices according descending lengths
    sorted_seqlens = sorted(seqlens, key=lambda x: x[0], reverse=True)

    seqs,lens = [],[]
    for lenz, oidx in sorted_seqlens:
        lens.append(lenz)
        seqs.append(inputs[oidx])

    return seqs,lens

def details(t,loss,acc,vloss,vacc):
    print("time cost:"+str(t/60)+" mins")
    print("valid ave_Loss: "+str(vloss)+" | average distance: "+str(vacc))
    print("train ave_Loss: "+str(loss)+" | average distance: "+str(acc))
    print(" ")
    print("---------------------------------------")

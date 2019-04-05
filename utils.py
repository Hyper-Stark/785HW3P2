import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence


def pad_at_end(pairs):

    if len(pairs[0]) == 2:
#        print("train mode")
        xs,ys,sortedxs = [],[],[]
        for i,(x,y) in enumerate(pairs):
            xs.append(x)
            ys.append(y)
            sortedxs.append((x.shape[0],i))
            
        xs = pad_sequence(xs,batch_first=True,padding_value=0)
        xs = xs.permute(0,2,1)

        labels = [None] * len(ys)
        result = torch.zeros(xs.shape)
        lengths = torch.zeros(xs.shape[0], dtype=torch.int)
        sortedxs = sorted(sortedxs, key = lambda X: X[0], reverse=True)
        for nidx,(lenz,oidx) in enumerate(sortedxs):
            result[nidx] = xs[oidx]
            labels[nidx] = ys[oidx].type(torch.IntTensor) + 1
            lengths[nidx] = lenz
        
        return result,labels,lengths

    else:

#        print("eval mode")
        xs,sortedxs = [],[]
        for i,x in enumerate(pairs):
            xs.append(x)
            sortedxs.append((x.shape[0],i))
            
        xs = pad_sequence(xs,batch_first=True,padding_value=0)
        xs = xs.permute(0,2,1)

        result = torch.zeros(xs.shape)
        lengths = torch.zeros(xs.shape[0], dtype=torch.int)
        sortedxs = sorted(sortedxs, key = lambda X: X[0], reverse=True)
        for nidx,(lenz,oidx) in enumerate(sortedxs):
            result[nidx] = xs[oidx]
            lengths[nidx] = lenz
        
        return result,lengths

       

def details(t,loss,acc,vloss,vacc):
    print("time cost:"+str(t/60)+" mins")
    print("valid ave_Loss: "+str(vloss)+" | average distance: "+str(vacc))
    print("train ave_Loss: "+str(loss)+" | average distance: "+str(acc))
    print(" ")
    print("---------------------------------------")


    

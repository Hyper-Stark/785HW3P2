import torch
import Levenshtein as L
import torch.nn.functional as F

from data.phoneme_list import *
from ctcdecode import CTCBeamDecoder

class Predictor(object):

    def __init__(self):
        super().__init__()
        self.labels = [' '] + PHONEME_MAP
        self.decoder = CTCBeamDecoder(
            labels=self.labels,
            beam_width=100,
            blank_id=0,
            num_processes=32
        )

    def __call__(self, logits, labels, label_lens):
        return self.forward(logits, labels, label_lens)

    def evaluateError(self, logits, labels, label_lens):

        logits = torch.transpose(logits, 0, 1).cpu()
        probs = F.softmax(logits, dim=2)
#        print("                begin to decode ctc")
        parse_res = self.decoder.decode(probs=probs)
        output, scores, timesteps, out_seq_len = parse_res

#        print("                begin to calculate distance")
        pos, ls = 0, 0.
        for i in range(output.size(0)):
            pred = "".join(self.labels[o] for o in output[i, 0, :out_seq_len[i, 0]])
            true = "".join(self.labels[l] for l in labels[pos:pos + label_lens[i]])
            #print("Pred: {}, True: {}".format(pred, true))
            pos += label_lens[i] 
            ls += L.distance(pred, true)

#        print("                finished decoding")
        assert pos == labels.size(0)
        return ls

    def predict(self, logits):
        logits = torch.transpose(logits, 0, 1).cpu()
        probs = F.softmax(logits, dim=2)
        parse_res = self.decoder.decode(probs=probs)
        output, scores, timesteps, out_seq_len = parse_res

        result = []
        for i in range(output.size(0)):
            pred = "".join(self.labels[o] for o in output[i, 0, :out_seq_len[i,0]])
            result.append(pred)

        return result

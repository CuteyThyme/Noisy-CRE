from sys import flags
import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import BertModel

from .backbone import Bert_Encoder

class Encoder(nn.Module):
    def __init__(self, args, hidden=False):
        super().__init__()

        self.encoder = Bert_Encoder(args,hidden=hidden)
        self.output_size = self.encoder.out_dim
        dim_in = self.output_size
        self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, args.feat_dim)
            )
        self.dropout = nn.Dropout(p=0.5)
        self.hidden = args.hidden
    def bert_forward(self, x, aug=False, flag=False, embedding=None):
        if self.hidden is False:
            out = self.encoder(x, flag=flag, embedding=embedding)
            aug_out = self.dropout(out) # aug represention
            xx = self.head(out)
            aug_xx = self.head(aug_out)
            xx = F.normalize(xx, p=2, dim=1)
            aug_xx = F.normalize(aug_xx, p=2, dim=1)
            if aug is False:
                return out, xx
            else:
                return out, xx, aug_xx
        else:
            out, hidden_states = self.encoder(x, flag=flag, embedding=embedding)
            aug_out = self.dropout(out) # aug represention
            xx = self.head(out)
            aug_xx = self.head(aug_out)
            xx = F.normalize(xx, p=2, dim=1)
            aug_xx = F.normalize(aug_xx, p=2, dim=1)
            if aug is False:
                return out, xx, hidden_states
            else:
                return out, xx, aug_xx, hidden_states

class BasicBertModel(nn.Module):
    def __init__(self, args, num_labels, hidden=False):
        super(BasicBertModel, self).__init__()
        self.encoder = Bert_Encoder(args,hidden=hidden)
        self.output_size = self.encoder.out_dim # 768
        self.classifier = nn.Linear(self.output_size, num_labels)
        self.hidden = hidden

    def forward(self, x, flag=False, embedding=None):
        if  self.hidden is False:
            outputs = self.encoder(x,flag=flag,embedding=embedding)
            logits = self.classifier(outputs)
            return logits
        else:
            outputs,hidden_states = self.encoder(x,flag=flag,embedding=embedding)
            logits = self.classifier(outputs)
            return logits,hidden_states
    def get_input_embeddings(self):
        return self.encoder.encoder.get_input_embeddings()

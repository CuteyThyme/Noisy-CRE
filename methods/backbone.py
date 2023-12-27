import torch.nn as nn
import torch
import numpy as np
from transformers import BertModel, BertConfig

class Bert_Encoder(nn.Module):

    def __init__(self, config, hidden=False):
        super(Bert_Encoder, self).__init__()
        
        self.gpu = config.gpu
        self.hidden = hidden
        # load model
        #self.encoder = BertModel.from_pretrained(config.bert_path).to(self.gpu)
        self.bert_config = BertConfig.from_pretrained(config.bert_path)
        #self.bert_config.output_hidden_states = self.hidden # set true to get hidden of 13 layers
        self.encoder = BertModel.from_pretrained(config.bert_path, output_hidden_states=self.hidden).cuda()
        # the dimension for the final outputs
        self.output_size = config.encoder_output_size
        self.out_dim = self.output_size

        # find which encoding is used
        if config.pattern in ['standard', 'entity_marker']:
            self.pattern = config.pattern
        else:
            raise Exception('Wrong encoding.')

        if self.pattern == 'entity_marker':
            self.encoder.resize_token_embeddings(config.vocab_size + config.marker_size)
            self.linear_transform = nn.Linear(self.bert_config.hidden_size*2, self.output_size, bias=True)
        else:
            self.linear_transform = nn.Linear(self.bert_config.hidden_size, self.output_size, bias=True)

        self.layer_normalization = nn.LayerNorm([self.output_size])


    def get_output_size(self):
        return self.output_size

    def forward(self, inputs, flag=False, embedding=None):
        # generate representation under a certain encoding strategy
        if self.pattern == 'standard':
            # in the standard mode, the representation is generated according to
            #  the representation of[CLS] mark.
            output = self.encoder(inputs)[1]
            return output
        else:
            # in the entity_marker mode, the representation is generated from the representations of
            #  marks [E11] and [E21] of the head and tail entities.
            e11 = []
            e21 = []
            # for each sample in the batch, acquire the positions of its [E11] and [E21]
            for i in range(inputs.size()[0]):
                tokens = inputs[i].cpu().numpy()
                e11.append(np.argwhere(tokens == 30522)[0][0])
                e21.append(np.argwhere(tokens == 30524)[0][0])

            # input the sample to BERT
            if self.hidden is False:
                if flag is False:
                    tokens_output = self.encoder(inputs)[0] # [B,N] --> [B,N,H]
                else:
                    print("using embedding...")
                    tokens_output = self.encoder(inputs_embeds=embedding)[0]
                output = []
                # for each sample in the batch, acquire its representations for [E11] and [E21]
                for i in range(len(e11)):
                    if inputs.device.type in ['cuda']:
                        instance_output = torch.index_select(tokens_output, 0, torch.tensor(i).to(self.gpu)) # 定位batch
                        instance_output = torch.index_select(instance_output, 1, torch.tensor([e11[i], e21[i]]).to(self.gpu)) # 定位e11,e21
                    else:
                        instance_output = torch.index_select(tokens_output, 0, torch.tensor(i))
                        instance_output = torch.index_select(instance_output, 1, torch.tensor([e11[i], e21[i]]))
                    output.append(instance_output) # [B,N] --> [B,2,H]
                # for each sample in the batch, concatenate the representations of [E11] and [E21], and reshape
                output = torch.cat(output, dim=0) # 把所有batch的[E11,E21]拼接起来
                output = output.view(output.size()[0], -1) # [B,N] --> [B,H*2]
                
                output = self.linear_transform(output)
                return output
            else:
                if flag is False:
                    tokens_outputs = self.encoder(inputs)[2][1:] # [B,N] --> 12*[B,N,H], 1~12 layers
                else:
                    tokens_outputs = self.encoder(inputs_embeds=embedding)[2][1:] # [B,N] --> 12*[B,N,H], 1~12 layers
                outputs = []
                for tokens_output in tokens_outputs:
                    output = []
                    # for each sample in the batch, acquire its representations for [E11] and [E21]
                    for i in range(len(e11)):
                        if inputs.device.type in ['cuda']:
                            instance_output = torch.index_select(tokens_output, 0, torch.tensor(i).to(self.gpu)) # 定位batch
                            instance_output = torch.index_select(instance_output, 1, torch.tensor([e11[i], e21[i]]).to(self.gpu)) # 定位e11,e21
                        else:
                            instance_output = torch.index_select(tokens_output, 0, torch.tensor(i))
                            instance_output = torch.index_select(instance_output, 1, torch.tensor([e11[i], e21[i]]))
                        output.append(instance_output) # [B,N] --> [B,2,H]
                    # for each sample in the batch, concatenate the representations of [E11] and [E21], and reshape
                    output = torch.cat(output, dim=0) # 把所有batch的[E11,E21]拼接起来
                    output = output.view(output.size()[0], -1) # [B,N] --> [B,H*2]
                    
                    output = self.linear_transform(output)
                    outputs.append(output)
                return output, outputs
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import time
import ipdb
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler







class GRUNet(nn.Module):

    def __init__(self, network_configs):

        super(GRUNet, self).__init__()

        self.coord_embed = nn.Linear(network_configs['coord_input_dim'], network_configs['embed_dim'], bias=False)
        self.feat_embed = nn.Embedding(network_configs['feat_dict_size'], network_configs['embed_dim'])
        self.hidden_size = network_configs['hidden_size']
        self.gru = nn.GRU(input_size= network_configs['embed_dim'], hidden_size= network_configs['hidden_size'], num_layers= network_configs['num_layers'], batch_first=True, dropout=network_configs['dropout'], bidirectional=True)
        self.out_layer = nn.Linear(network_configs['hidden_size'] * 2, network_configs['num_classes'])

    def forward(self, coordinate, flag_bits, position_encoding):

        x = self.coord_embed(coordinate) + self.feat_embed(flag_bits) + self.feat_embed(position_encoding)


        
        self.rnn_hidden_feature, h = self.gru(x)
        featur = torch.cat(( self.rnn_hidden_feature[:, -1, : self.hidden_size], self.rnn_hidden_feature[:, -1, self.hidden_size: ]), 1)
        x = self.out_layer(featur)
        return x, featur

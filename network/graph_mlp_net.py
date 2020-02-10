import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters 
        # with bias 0 and weight unif(0,1) which is too large!
        self.init_parameters()

    def init_parameters(self):
        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input, mask=None):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class GraphMLPLayer(nn.Module):

    def __init__(self, embed_dim, dropout=0.1, normalization='batch'):
        super(GraphMLPLayer, self).__init__()
        
        self.sub_layers = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.norm = Normalization(embed_dim, normalization)
        
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
    
    def forward(self, input):
        return self.norm(self.sub_layers(input))


class GraphMLPEncoder(nn.Module):
    
    def __init__(self, coord_input_dim, feat_input_dim, feat_dict_size, 
                 n_layers=3, embed_dim=256, dropout=0.1):         
        
        super(GraphMLPEncoder, self).__init__()
        
        # Embedding/Input layers
        self.coord_embed = nn.Linear(coord_input_dim, embed_dim, bias=False)
        self.feat_embed = nn.Embedding(feat_dict_size, embed_dim)
        
        # MLP blocks
        self.mlp_layers = nn.ModuleList([
            GraphMLPLayer(embed_dim * 3, dropout) 
                for _ in range(n_layers)
        ])
        
        self.init_parameters()
        
    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, coord, flag, pos):
        
        # Embed inputs to embed_dim
        h = torch.cat((self.coord_embed(coord), self.feat_embed(flag), self.feat_embed(pos)), dim=2)
        
        # Perform n_layers of Graph MLP blocks
        for layer in self.mlp_layers:
            h = layer(h)
        
        return h


class GraphMLPClassifier(nn.Module):
    
    def __init__(self, n_classes, coord_input_dim, feat_input_dim, feat_dict_size, 
                 n_layers=3, embed_dim=256, feedforward_dim=1024, dropout=0.1):
        
        super(GraphMLPClassifier, self).__init__()
        
        self.encoder = GraphMLPEncoder(
            coord_input_dim, feat_input_dim, feat_dict_size, 
            n_layers, embed_dim, dropout)
        
        self.mlp_classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, feedforward_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, feedforward_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, n_classes, bias=True)
        )
    
    def forward(self, coord, flag, pos, attention_mask=None, padding_mask=None, true_seq_length=None):
        """
        Args:
            coord: Input coordinates (batch_size, seq_length, coord_input_dim)
            # TODO feat: Input features (batch_size, seq_length, feat_input_dim)
            attention_mask: Masks for attention computation (batch_size, seq_length, seq_length)
                            Attention mask should contain -inf if attention is not possible 
                            (i.e. mask is a negative adjacency matrix)
            padding_mask: Mask indicating padded elements in input (batch_size, seq_length)
                          Padding mask element should be 1 if valid element, 0 if padding
                          (i.e. mask is a boolean multiplicative mask)
            true_seq_length: True sequence lengths for input (batch_size, )
                             Used for computing true mean of node embeddings for graph embedding
        
        Returns:
            logits: Un-normalized logits for class prediction (batch_size, n_classes)
        """
        
        # Embed input sequence
        h = self.encoder(coord, flag, pos)
                
        # Mask out padding embeddings to zero
        if padding_mask is not None:
            masked_h = h * padding_mask.type_as(h)
            g = masked_h.sum(dim = 1)
            # g = masked_h.sum(dim=1)/true_seq_length.type_as(h)
            
        else:
            g = h.sum(dim=1)
        
        # Compute logits
        logits = self.mlp_classifier(g)
        
        return logits
 
    
def make_model(n_classes=345, coord_input_dim=2, feat_input_dim=2, feat_dict_size=104, 
               n_layers=3, n_heads=8, embed_dim=256, feedforward_dim=1024, 
               normalization='batch', dropout=0.1):
    
    model = GraphMLPClassifier(
        n_classes, coord_input_dim, feat_input_dim, feat_dict_size, n_layers, 
        embed_dim, feedforward_dim, dropout)
    
    print(model)
    nb_param = 0
    for param in model.parameters():
        nb_param += np.prod(list(param.data.size()))
    print('Number of parameters: ', nb_param)

    return model

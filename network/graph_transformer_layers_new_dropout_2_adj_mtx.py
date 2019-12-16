import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
import math
import ipdb



class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input, mask=None):
        return input + self.module(input, mask=mask)
    

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


class MultiHeadAttention(nn.Module):
    
    def __init__(self, n_heads, input_dim, embed_dim=None, 
                 val_dim=None, key_dim=None, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))
            
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
        Args:
            q: Input queries (batch_size, n_query, input_dim)
            h: Input data (batch_size, graph_size, input_dim)
            mask: Input attention mask (batch_size, n_query, graph_size)
                  or viewable as that (i.e. can be 2 dim if n_query == 1);
                  Mask should contain -inf if attention is not possible 
                  (i.e. mask is a negative adjacency matrix)
        
        Returns: 
            out: Updated data after attention (batch_size, graph_size, input_dim)
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        dropt1_qflat = self.dropout_1(qflat)
        Q = torch.matmul(dropt1_qflat, self.W_query).view(shp_q)

        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        dropt2_hflat = self.dropout_2(hflat)
        K = torch.matmul(dropt2_hflat, self.W_key).view(shp)

        dropt3_hflat = self.dropout_3(hflat)
        V = torch.matmul(dropt3_hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        
        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility = compatibility + mask.type_as(compatibility)

        attn = F.softmax(compatibility, dim=-1)

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)
        
        #out = self.drop(out)

        return out
        

class PositionWiseFeedforward(nn.Module):
    
    def __init__(self, embed_dim, feedforward_dim=512, dropout=0.1):
        super(PositionWiseFeedforward, self).__init__()
        # modified on 2019 10 23
        self.sub_layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim, bias=True),
            nn.ReLU()
        )
        
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
    
    def forward(self, input, mask=None):
        return self.sub_layers(input)


class MultiGraphTransformerLayer(nn.Module):

    def __init__(self, n_heads, embed_dim, feedforward_dim, 
                 normalization='batch', dropout=0.1):
        super(MultiGraphTransformerLayer, self).__init__()
        
        self.self_attention1 = SkipConnection(
            MultiHeadAttention(
                    n_heads=n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim,
                    dropout=dropout
                )
            )
        self.self_attention2 = SkipConnection(
            MultiHeadAttention(
                    n_heads=n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim,
                    dropout=dropout
                )
            )

        # modified on 2019 10 24.
        self.tmp_linear_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim, bias=True),
            nn.ReLU(),
        )

        self.norm1 = Normalization(embed_dim, normalization)
        
        self.positionwise_ff = SkipConnection(
               PositionWiseFeedforward(
                   embed_dim=embed_dim,
                   feedforward_dim=feedforward_dim,
                   dropout=dropout
                )
            )
        self.norm2 = Normalization(embed_dim, normalization)
        
    def forward(self, input, mask1, mask2):
        #ipdb.set_trace()
        h1 = self.self_attention1(input, mask=mask1)
        h2 = self.self_attention2(input, mask=mask2)
        hh = torch.cat((h1, h2), dim=2)
        hh = self.tmp_linear_layer(hh)
        #ipdb.set_trace()
        hh = self.norm1(hh, mask=mask1)
        hh = self.positionwise_ff(hh, mask=mask1)
        hh = self.norm2(hh, mask=mask1)
        return hh
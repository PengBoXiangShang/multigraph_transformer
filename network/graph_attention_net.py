import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph_transformer_layers_new_dropout import *


class GraphAttentionLayer(nn.Module):

    def __init__(self, n_heads, embed_dim, feedforward_dim, 
                 normalization='batch', dropout=0.1):
        super(GraphAttentionLayer, self).__init__()
        
        self.self_attention = SkipConnection(
            MultiHeadAttention(
                    n_heads=n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim,
                    dropout=dropout
                )
            )
        self.norm = Normalization(embed_dim, normalization)
        
    def forward(self, input, mask):
        h = F.relu(self.self_attention(input, mask=mask))
        h = self.norm(h, mask=mask)
        return h


class GraphAttentionEncoder(nn.Module):
    
    def __init__(self, coord_input_dim, feat_input_dim, feat_dict_size, n_layers=3, n_heads=8, 
                 embed_dim=256, feedforward_dim=1024, normalization='batch', dropout=0.1):         
        
        super(GraphAttentionEncoder, self).__init__()
        
        # Embedding/Input layers
        self.coord_embed = nn.Linear(coord_input_dim, embed_dim, bias=False)
        self.feat_embed = nn.Embedding(feat_dict_size, embed_dim)
        
        # Transformer blocks
        self.attention_layers = nn.ModuleList([
            GraphAttentionLayer(n_heads, embed_dim * 3, feedforward_dim, normalization, dropout) 
                for _ in range(n_layers)
        ])

    def forward(self, coord, flag, pos, attention_mask=None):
        
        # Embed inputs to embed_dim
        h = torch.cat((self.coord_embed(coord), self.feat_embed(flag), self.feat_embed(pos)), dim=2)
        
        # Perform n_layers of Graph Attention blocks
        for layer in self.attention_layers:
            h = layer(h, mask=attention_mask)
        
        return h


class GraphAttentionClassifier(nn.Module):
    
    def __init__(self, n_classes, coord_input_dim, feat_input_dim, feat_dict_size, 
                 n_layers=3, n_heads=8, embed_dim=256, feedforward_dim=1024, 
                 normalization='batch', dropout=0.1):
        
        super(GraphAttentionClassifier, self).__init__()
        
        self.encoder = GraphAttentionEncoder(
            coord_input_dim, feat_input_dim, feat_dict_size, n_layers, 
            n_heads, embed_dim, feedforward_dim, normalization, dropout)
        
        self.mlp_classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, feedforward_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, feedforward_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, n_classes, bias=True)
        )
    
    def forward(self, coord, flag, pos, attention_mask=None,
                padding_mask=None, true_seq_length=None):
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
        h = self.encoder(coord, flag, pos, attention_mask)
                
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
    
    model = GraphAttentionClassifier(
        n_classes, coord_input_dim, feat_input_dim, feat_dict_size, n_layers, 
        n_heads, embed_dim, feedforward_dim, normalization, dropout)
    
    print(model)
    nb_param = 0
    for param in model.parameters():
        nb_param += np.prod(list(param.data.size()))
    print('Number of parameters: ', nb_param)

    return model

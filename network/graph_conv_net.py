import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
import math


class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input, mask=None):
        return input + self.module(input, mask=mask)
    

class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='layer'):
        super(Normalization, self).__init__()

        self.normalizer = {
            'layer': nn.LayerNorm(embed_dim, )
            'batch': nn.BatchNorm1d(embed_dim, affine=True, track_running_stats=True),
            'instance': nn.InstanceNorm1d(embed_dim, affine=True, track_running_stats=True)
        }.get(normalization, None)

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


class NodeFeatures(nn.Module):
    """Convnet features for nodes
    """
    
    def __init__(self, embed_dim, normalization='batch', dropout=0.1):
        super(NodeFeatures, self).__init__()
        self.U = nn.Linear(embed_dim, embed_dim, True)
        self.V = nn.Linear(embed_dim, embed_dim, True)
        self.drop = nn.Dropout(dropout)
        
        self.init_parameters()
        
    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, x, mask=None):
        """
        Args:
            x: Node features (batch_size, num_nodes, embed_dim)
        Returns:
            x_new: Convolved node features (batch_size, num_nodes, embed_dim)
        """
        num_nodes, embed_dim = x.shape[1], x.shape[2]
        
        Ux = self.U(x)  # B x V x H
        Vx = self.V(x)  # B x V x H
        
        # extend Vx from "B x V x H" to "B x V x V x H"
        Vx = Vx.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            Vx = Vx * mask.type_as(Vx)
            
        x_new = Ux + torch.sum(Vx, dim=2)  # B x V x H
        
        x_new = F.relu(x_new)
        x_new = self.drop(x_new)
        
        return x_new


class GraphConvNetLayer(nn.Module):
    """Graph Convnet layer
    """

    def __init__(self, embed_dim, normalization='batch', dropout=0.1):
        super(GraphConvNetLayer, self).__init__()
        self.node_feat = SkipConnection(
            NodeFeatures(embed_dim, normalization, dropout)
        )
        self.norm = Normalization(embed_dim, normalization)

    def forward(self, x, mask=None):
        """
        Args:
            x: Node features (batch_size, num_nodes, embed_dim)
        """
        return self.norm(self.node_feat(x, mask=mask))



class GraphConvNetEncoder(nn.Module):
    
    def __init__(self, coord_input_dim, feat_input_dim, feat_dict_size, n_layers=3,
                 embed_dim=256, normalization='batch', dropout=0.1):         
        
        super(GraphConvNetEncoder, self).__init__()
        
        # Embedding/Input layers
        self.coord_embed = nn.Linear(coord_input_dim, embed_dim, bias=False)
        self.feat_embed = nn.Embedding(feat_dict_size, embed_dim)
        
        # GCN blocks
        self.gcn_layers = nn.ModuleList([
            GraphConvNetLayer(embed_dim * 3, normalization, dropout) 
                for _ in range(n_layers)
        ])

    def forward(self, coord, flag, pos, attention_mask=None, padding_mask=None):
        
        # Embed inputs to embed_dim
        h = torch.cat((self.coord_embed(coord), self.feat_embed(flag), self.feat_embed(pos)), dim=2)
        
        # Perform n_layers of Graph ConvNet blocks
        for layer in self.gcn_layers:
            # Mask out padding embeddings to zero
            if padding_mask is not None:
                h = h * padding_mask.type_as(h)
            
            h = layer(h, mask=attention_mask)
        
        return h


class GraphConvNetClassifier(nn.Module):
    
    def __init__(self, n_classes, coord_input_dim, feat_input_dim, feat_dict_size, 
                 n_layers=3, embed_dim=256, feedforward_dim=1024, 
                 normalization='batch', dropout=0.1):
        
        super(GraphConvNetClassifier, self).__init__()
        
        self.encoder = GraphConvNetEncoder(
            coord_input_dim, feat_input_dim, feat_dict_size, 
            n_layers, embed_dim, normalization, dropout)
        
        self.mlp_classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, feedforward_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, feedforward_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, n_classes, bias=True)
        )

    def _mask_to_adj(A):
        """Helper function to convert adjacency mask into adjacency matrix format.
        1 --> connection, 0 --> no connection
        """
        A[A==0] = 1
        A[A<0] = -1e10
        return A

    
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
        if attention_mask is not None:
            attention_mask = self._mask_to_adj(attention_mask)

        # Embed input sequence
        h = self.encoder(coord, flag, pos, attention_mask, padding_mask)
                
        # Mask out padding embeddings to zero
        if padding_mask is not None:
            masked_h = h * padding_mask.type_as(h)
            g = masked_h.sum(dim=1)
            # g = masked_h.sum(dim=1)/true_seq_length.type_as(h)
            
        else:
            g = h.sum(dim=1)
        
        # Compute logits
        logits = self.mlp_classifier(g)
        
        return logits
 
    
def make_model(n_classes=345, coord_input_dim=2, feat_input_dim=2, feat_dict_size=104, 
               n_layers=3, n_heads=8, embed_dim=256, feedforward_dim=1024, 
               normalization='batch', dropout=0.1):
    
    model = GraphConvNetClassifier(
        n_classes, coord_input_dim, feat_input_dim, feat_dict_size, n_layers, 
        embed_dim, feedforward_dim, normalization, dropout)
    
    print(model)
    nb_param = 0
    for param in model.parameters():
        nb_param += np.prod(list(param.data.size()))
    print('Number of parameters: ', nb_param)

    return model


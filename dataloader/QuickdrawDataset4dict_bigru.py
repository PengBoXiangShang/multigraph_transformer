import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import time




def generate_attention_mask(stroke_length):
    attention_mask = np.zeros([100, 100], int)
    attention_mask[stroke_length: , :] = -1e8
    attention_mask[:, stroke_length : ] = -1e8
    return attention_mask



def generate_padding_mask(stroke_length):
    padding_mask = np.ones([100, 1], int)
    padding_mask[stroke_length: , : ] = 0
    return padding_mask



class QuickdrawDataset(data.Dataset):

    def __init__(self, coordinate_path_root, sketch_list, data_dict):
        with open(sketch_list) as sketch_url_file:
            sketch_url_list = sketch_url_file.readlines()
            
            self.coordinate_urls = [os.path.join(coordinate_path_root, (sketch_url.strip(
            ).split(' ')[0]).replace('png', 'npy')) for sketch_url in sketch_url_list]
            
            self.labels = [int(sketch_url.strip().split(' ')[-1])
                           for sketch_url in sketch_url_list]

            self.data_dict = data_dict
        

    def __len__(self):
        return len(self.coordinate_urls)

    def __getitem__(self, item):
        
        
        coordinate_url = self.coordinate_urls[item]
        label = self.labels[item]
        
        
        coordinate, flag_bits, stroke_len = self.data_dict[coordinate_url]

    
        attention_mask = generate_attention_mask(stroke_len)
        padding_mask = generate_padding_mask(stroke_len)

        position_encoding = np.arange(100)
        position_encoding.resize([100, 1])

        if coordinate.dtype == 'object':
            coordinate = coordinate[0]
        
        assert coordinate.shape == (100, 2)
        
        coordinate = coordinate.astype('float32') 
        
        return (coordinate, label, flag_bits.astype('int'), stroke_len, attention_mask, padding_mask, position_encoding)

    

import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import time
import numpy as np



def produce_adjacent_matrix_random(stroke_length, conn=0.15):
    attention_mask = np.random.choice(a=[0, -1e8], size=[100, 100], p=[conn, 1-conn])
    attention_mask[stroke_length: , :] = -1e8
    attention_mask[:, stroke_length : ] = -1e8
    #####
    attention_mask = np.triu(attention_mask)
    attention_mask += attention_mask.T - np.diag(attention_mask.diagonal())
    
    for i in range(stroke_length):
        attention_mask[i, i] = 0
    return attention_mask



def check_adjacent_matrix(adjacent_matrix, stroke_len):
    assert adjacent_matrix.shape == (100, 100)
    for idx in range(1, stroke_len):
        assert adjacent_matrix[idx][idx - 1] == adjacent_matrix[idx - 1][idx]



def generate_padding_mask(stroke_length):
    padding_mask = np.ones([100, 1], int)
    padding_mask[stroke_length: , : ] = 0
    return padding_mask



class QuickdrawDataset_random_attmask(data.Dataset):

    def __init__(self, coordinate_path_root, sketch_list, data_dict, non_zero_ratio):
        with open(sketch_list) as sketch_url_file:
            sketch_url_list = sketch_url_file.readlines()
            
            self.coordinate_urls = [os.path.join(coordinate_path_root, (sketch_url.strip(
            ).split(' ')[0]).replace('png', 'npy')) for sketch_url in sketch_url_list]
            
            self.labels = [int(sketch_url.strip().split(' ')[-1])
                           for sketch_url in sketch_url_list]

            self.data_dict = data_dict

            self.non_zero_ratio = non_zero_ratio
        

    def __len__(self):
        return len(self.coordinate_urls)

    def __getitem__(self, item):
        
        
        coordinate_url = self.coordinate_urls[item]
        label = self.labels[item]
        
        #
        coordinate, flag_bits, stroke_len = self.data_dict[coordinate_url]

        #
        
        attention_mask = produce_adjacent_matrix_random(stroke_len, self.non_zero_ratio)
        check_adjacent_matrix(attention_mask, stroke_len)

        padding_mask = generate_padding_mask(stroke_len)

        position_encoding = np.arange(100)
        position_encoding.resize([100, 1])

        if coordinate.dtype == 'object':
            coordinate = coordinate[0]
        
        assert coordinate.shape == (100, 2)
        
        coordinate = coordinate.astype('float32') 
        # print(type(coordinate), type(label), type(flag_bits), type(stroke_len), type(attention_mask), type(padding_mask), type(position_encoding))
        return (coordinate, label, flag_bits.astype('int'), stroke_len, attention_mask, padding_mask, position_encoding)

    

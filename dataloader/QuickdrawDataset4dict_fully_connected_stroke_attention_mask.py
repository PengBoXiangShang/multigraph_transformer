import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import time




def detect_ending_points(flag_bits, stroke_len):

    assert flag_bits.shape == (100, 1)
    ending_points_index_array = []
    # 
    ending_points_index_array.append(-1)
    
    for idx in range(stroke_len):
        # 
        if (flag_bits[idx] == 101):
            ending_points_index_array.append(idx)

    # 
    if stroke_len == 100:

        assert flag_bits[99] == 100 or flag_bits[99] == 101

        if flag_bits[99] == 100:
            ending_points_index_array.append(99)

    return ending_points_index_array

def produce_adjacent_matrix(ending_points_index_array):
    adja_matr = np.zeros([100, 100], int)
    
    adja_matr[ : ][ : ] = -1e8
    for idx in range(1, len(ending_points_index_array)):
        start_index = ending_points_index_array[idx - 1] + 1
        end_index = ending_points_index_array[idx]
        # 
        if end_index == 99:
            adja_matr[start_index : , start_index : ] = 0
        else:
            adja_matr[start_index : end_index + 1, start_index: end_index + 1] = 0

    return adja_matr


def generate_padding_mask(stroke_length):
    padding_mask = np.ones([100, 1], int)
    padding_mask[stroke_length: , : ] = 0
    return padding_mask



class QuickdrawDataset_fully_connected_stroke_attmask(data.Dataset):

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
        
        #
        coordinate, flag_bits, stroke_len = self.data_dict[coordinate_url]

        #
        ending_points_index_array = detect_ending_points(flag_bits, stroke_len)
        attention_mask = produce_adjacent_matrix(ending_points_index_array)

        padding_mask = generate_padding_mask(stroke_len)

        position_encoding = np.arange(100)
        position_encoding.resize([100, 1])

        if coordinate.dtype == 'object':
            coordinate = coordinate[0]
        
        assert coordinate.shape == (100, 2)
        
        coordinate = coordinate.astype('float32') 
        # print(type(coordinate), type(label), type(flag_bits), type(stroke_len), type(attention_mask), type(padding_mask), type(position_encoding))
        return (coordinate, label, flag_bits.astype('int'), stroke_len, attention_mask, padding_mask, position_encoding)

    

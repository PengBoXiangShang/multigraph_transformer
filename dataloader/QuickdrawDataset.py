import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import time




class QuickdrawDataset(data.Dataset):

    def __init__(self, sketch_path_root, sketch_list, data_transforms=None):

        with open(sketch_list) as sketch_url_file:
            sketch_url_list = sketch_url_file.readlines()
            self.sketch_urls = [os.path.join(sketch_path_root, sketch_url.strip().split(' ')[
                                             0]) for sketch_url in sketch_url_list]
            
            self.labels = [int(sketch_url.strip().split(' ')[-1])
                           for sketch_url in sketch_url_list]


        self.data_transforms = data_transforms

    def __len__(self):
        return len(self.sketch_urls)

    def __getitem__(self, item):

        sketch_url = self.sketch_urls[item]

        label = self.labels[item]
        
        
        sketch = Image.open(sketch_url, 'r')
        

        if self.data_transforms is not None:
            try:
                sketch = self.data_transforms(sketch)
            except:
                print("Cannot transform sketch: {}".format(sketch_url))

        return sketch, label

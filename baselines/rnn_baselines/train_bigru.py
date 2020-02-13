import argparse
import collections
import datetime
#import imp
import os
import pickle
import time
#import lmdb
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
sys.path.append("../..")
from utils.AverageMeter import AverageMeter
from utils.Logger import Logger
from utils.accuracy import *
from dataloader.QuickdrawDataset4dict_bigru import *
from network.Bidirectional_GRU import *





################################################
# This python file contains five parts:
#
# Part 1. Argument Parser
# Part 2. configurations:
#                       Part 2-1. Basic configuration
#                       Part 2-2. dataloader instantiation
#                       Part 2-3. log configuration
#                       Part 2-4. configurations for loss function, network, and optimizer
# Part 3. 'train' function
# Part 4. 'validate' function 
# Part 5. 'main' function
################################################


# Part 1. Argument Parser
parser = argparse.ArgumentParser(description='bigru_based_sketch_classifiction')
parser.add_argument("--exp", type=str, default="bigru_001", help="experiment")
# TODO
parser.add_argument("--train_coordinate_path_root", type=str, default="/home/peng/dataset/tiny_quickdraw_coordinate/train/", help="train_sketch_coordinate_dir")
parser.add_argument("--val_coordinate_path_root", type=str, default="/home/peng/dataset/tiny_quickdraw_coordinate/val/", help="val_sketch_coordinate_dir")
parser.add_argument("--sketch_list", type=str, default="../../dataloader/tiny_train_set.txt", help="sketch_list_urls")
parser.add_argument("--sketch_list_4_val", type=str, default="../../dataloader/tiny_val_set.txt", help="sketch_list_urls_4_validation")
parser.add_argument("--batch_size", type=int, default=256, help="batch_size")
parser.add_argument("--num_workers", type=int, default=12, help="num_workers")
parser.add_argument('--gpu', type=str, default="0", help='choose GPU')


args = parser.parse_args()


# Part 2. configurations

# Part 2-1. Basic configuration
basic_configs = collections.OrderedDict()
basic_configs['serial_number'] = args.exp
basic_configs['learning_rate'] = 1e-3
basic_configs['num_epochs'] = 1000
basic_configs["lr_protocol"] = [(30, 1e-3), (60, 0.001 * 0.9), (90, 0.001 * (0.9 ** 2)), (120, 0.001 * (0.9 ** 3)), (150, 0.001 * (0.9 ** 4)), (180, 0.001 * (0.9 ** 5)), (210, 1e-4), (240, 1e-5)]
basic_configs["display_step"] = 10
lr_protocol = basic_configs["lr_protocol"]


# Part 2-2. dataloader instantiation
dataloader_configs = collections.OrderedDict()
dataloader_configs['train_coordinate_path_root'] = args.train_coordinate_path_root
dataloader_configs['val_coordinate_path_root'] = args.val_coordinate_path_root
dataloader_configs['sketch_list'] = args.sketch_list
dataloader_configs['sketch_list_4_val'] = args.sketch_list_4_val
dataloader_configs['batch_size'] = args.batch_size
dataloader_configs['num_workers'] = args.num_workers

dataloader_configs['data_dict_4_train'] = '../../dataloader/tiny_train_dataset_dict.pickle'
data_dict_4_train_f = open(dataloader_configs['data_dict_4_train'], 'rb')
data_dict_4_train = pickle.load(data_dict_4_train_f)  

dataloader_configs['data_dict_4_validation'] = '../../dataloader/tiny_val_dataset_dict.pickle' 
data_dict_4_validation_f = open(dataloader_configs['data_dict_4_validation'], 'rb')
data_dict_4_validation = pickle.load(data_dict_4_validation_f)  

#ipdb.set_trace()


# create dataset
# -----------------------------------------------------------------------------------------------------

train_dataset = QuickdrawDataset(dataloader_configs['train_coordinate_path_root'], dataloader_configs['sketch_list'], data_dict_4_train)
train_loader = DataLoader(train_dataset, batch_size=dataloader_configs['batch_size'], shuffle=True, num_workers=dataloader_configs['num_workers'])

val_dataset = QuickdrawDataset(dataloader_configs['val_coordinate_path_root'], dataloader_configs['sketch_list_4_val'], data_dict_4_validation)
val_loader = DataLoader(val_dataset, batch_size=dataloader_configs['batch_size'], shuffle=False, num_workers=dataloader_configs['num_workers'])

# Part 2-3. log configuration
exp_dir = os.path.join('./experimental_results', args.exp)

exp_log_dir = os.path.join(exp_dir, "log")
if not os.path.exists(exp_log_dir):
    os.makedirs(exp_log_dir)

exp_visual_dir = os.path.join(exp_dir, "visual")
if not os.path.exists(exp_visual_dir):
    os.makedirs(exp_visual_dir)

exp_ckpt_dir = os.path.join(exp_dir, "checkpoints")
if not os.path.exists(exp_ckpt_dir):
    os.makedirs(exp_ckpt_dir)

now_str = datetime.datetime.now().__str__().replace(' ', '_')
writer_path = os.path.join(exp_visual_dir, now_str)
writer = SummaryWriter(writer_path)

logger_path = os.path.join(exp_log_dir, now_str + ".log")
logger = Logger(logger_path).get_logger()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

logger.info("basic configuration settings: {}".format(basic_configs))


# Part 2-4. configurations for loss function, and optimizer

loss_function = nn.CrossEntropyLoss()

network_configs=collections.OrderedDict()
network_configs['coord_input_dim']= 2
network_configs['embed_dim']= 256
network_configs['feat_dict_size']= 103
network_configs['hidden_size']= 256
network_configs['num_layers']= 5
network_configs['dropout'] = 0.5
network_configs['num_classes']= 345


logger.info("network configuration settings: {}".format(network_configs))

net = GRUNet(network_configs)
net = net.cuda()


optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)


# Part 3. 'train' function
def train_function(epoch):
    training_loss = AverageMeter()
    training_acc = AverageMeter()
    net.train()

    lr = next((lr for (max_epoch, lr) in lr_protocol if max_epoch > epoch), lr_protocol[-1][1])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    logger.info("set learning rate to: {}".format(lr))
    
    for idx, (coordinate, label, flag_bits, _, _, _, position_encoding) in enumerate(tqdm(train_loader, ascii=True)):

        coordinate = coordinate.cuda()
        label = label.cuda()
        flag_bits = flag_bits.cuda()
        position_encoding = position_encoding.cuda()
        
        
        flag_bits.squeeze_(2)
        position_encoding.squeeze_(2)
     
        optimizer.zero_grad()

        output, _ = net(coordinate, flag_bits, position_encoding)
        # ipdb.set_trace()

    
        batch_loss = loss_function(output, label)

        batch_loss.backward()

        optimizer.step()

        training_loss.update(batch_loss.item(), coordinate.size(0))
        
        training_acc.update(accuracy(output, label, topk = (1,))[0].item(), coordinate.size(0))

        if (idx + 1) % basic_configs["display_step"] == 0:
            logger.info(
                "==> Iteration [{}][{}/{}]:".format(epoch + 1, idx + 1, len(train_loader)))
            logger.info("current batch loss: {}".format(
                batch_loss.item()))
            logger.info("average loss: {}".format(
                training_loss.avg))
            logger.info("average acc: {}".format(training_acc.avg))  

    
    logger.info("Begin Evaluating")

    validation_loss, validation_acc = validate_function()
    
    writer.add_scalars("loss", {
        "training_loss":training_loss.avg,
        "validation_loss":validation_loss.avg
        }, epoch+1)
    writer.add_scalars("acc", {
        "training_acc":training_acc.avg,
        "validation_acc":validation_acc.avg
        }, epoch+1)


# Part 4. 'validate' function
def validate_function():
    validation_loss = AverageMeter()
    validation_acc = AverageMeter()
    net.eval()
      
    with torch.no_grad():
        for idx, (coordinate, label, flag_bits, _, _, _, position_encoding) in enumerate(tqdm(val_loader, ascii=True)):
            
            coordinate = coordinate.cuda()
            label = label.cuda()
            flag_bits = flag_bits.cuda()
            position_encoding = position_encoding.cuda()

            
            flag_bits.squeeze_(2)
            position_encoding.squeeze_(2)
            
            output, _ = net(coordinate, flag_bits, position_encoding)

            batch_loss = loss_function(output, label)

            validation_loss.update(batch_loss.item(), coordinate.size(0))
            
            validation_acc.update(accuracy(output, label, topk = (1,))[0].item(), coordinate.size(0))

        logger.info("==> Evaluation Result: ")
        logger.info("loss: {}  acc:{}".format(validation_loss.avg, validation_acc.avg))

    return validation_loss, validation_acc





# Part 5. 'main' function
if __name__ == '__main__':

    logger.info("Begin Evaluating before training")
    validate_function()
    
    
    logger.info("training status: ")
    for epoch in range(basic_configs['num_epochs']):
        logger.info("Begin training epoch {}".format(epoch + 1))
        train_function(epoch)

        net_checkpoint_name = args.exp + "_net_epoch" + str(epoch + 1)
        net_checkpoint_path = os.path.join(exp_ckpt_dir, net_checkpoint_name)
        net_state = {"epoch": epoch + 1,
                     "network": net.state_dict()}
        torch.save(net_state, net_checkpoint_path)

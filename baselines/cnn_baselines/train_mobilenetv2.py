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
import torchvision.models as models
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
sys.path.append("../..")
from dataloader.QuickdrawDataset import *
from utils.AverageMeter import AverageMeter
from utils.Logger import Logger
from utils.accuracy import *









################################################
# This python file contains four parts:
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
parser = argparse.ArgumentParser(description='mobilenetv2_based_sketch_classifiction')
parser.add_argument("--exp", type=str, default="mobilenetv2_001", help="experiment")
# TODO
parser.add_argument("--train_picture_path_root", type=str, default="../../dataloader/data_4_cnnbaselines/tiny_train_set/", help="train_sketch_picture_dir")
parser.add_argument("--val_picture_path_root", type=str, default="../../dataloader/data_4_cnnbaselines/tiny_val_set/", help="val_sketch_picture_dir")
parser.add_argument("--test_picture_path_root", type=str, default="../../dataloader/data_4_cnnbaselines/tiny_test_set/", help="test_sketch_picture_dir")
parser.add_argument("--sketch_list", type=str, default="../../dataloader/data_4_cnnbaselines/tiny_train_set.txt", help="sketch_list_urls")
parser.add_argument("--sketch_list_4_val", type=str, default="../../dataloader/data_4_cnnbaselines/tiny_val_set.txt", help="sketch_list_urls_4_validation")
parser.add_argument("--sketch_list_4_test", type=str, default="../../dataloader/data_4_cnnbaselines/tiny_test_set.txt", help="sketch_list_urls_4_test")
parser.add_argument("--batch_size", type=int, default=128, help="batch_size")
parser.add_argument("--num_workers", type=int, default=8, help="num_workers")
parser.add_argument('--gpu', type=str, default="1", help='choose GPU')
# parser.add_argument('--gpu', type=str, default="0,1,2,3", help='choose GPU')


args = parser.parse_args()


# Part 2. configurations

# Part 2-1. Basic configuration
basic_configs = collections.OrderedDict()
basic_configs['serial_number'] = args.exp
basic_configs['learning_rate'] = 1e-3
basic_configs['num_epochs'] = 50
#basic_configs["lr_protocol"] = [(30, 1e-2), (60, 1e-3), (90, 1e-4), (120, 1e-5)]
basic_configs["lr_protocol"] = [(10, 1e-3), (20, (1e-3)*0.5), (30, (1e-3)*0.5*0.5), (40, (1e-3)*0.5*0.5*0.5), (50, (1e-3)*0.5*0.5*0.5*0.5)]
basic_configs["display_step"] = 10
lr_protocol = basic_configs["lr_protocol"]


# Part 2-2. dataloader instantiation
dataloader_configs = collections.OrderedDict()
dataloader_configs['train_picture_path_root'] = args.train_picture_path_root
dataloader_configs['val_picture_path_root'] = args.val_picture_path_root
dataloader_configs['test_picture_path_root'] = args.test_picture_path_root
dataloader_configs['sketch_list'] = args.sketch_list
dataloader_configs['sketch_list_4_val'] = args.sketch_list_4_val
dataloader_configs['sketch_list_4_test'] = args.sketch_list_4_test
dataloader_configs['batch_size'] = args.batch_size
dataloader_configs['num_workers'] = args.num_workers
 

transform_train = transforms.Compose([
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

transform_val = transforms.Compose([
    transforms.ToTensor()
])

#ipdb.set_trace()


# create dataset
# -----------------------------------------------------------------------------------------------------

train_dataset = QuickdrawDataset(dataloader_configs['train_picture_path_root'], dataloader_configs['sketch_list'], transform_train)
train_loader = DataLoader(train_dataset, batch_size=dataloader_configs['batch_size'], shuffle=True, num_workers=dataloader_configs['num_workers'])

val_dataset = QuickdrawDataset(dataloader_configs['val_picture_path_root'], dataloader_configs['sketch_list_4_val'], transform_val)
val_loader = DataLoader(val_dataset, batch_size=dataloader_configs['batch_size'], shuffle=False, num_workers=dataloader_configs['num_workers'])

test_dataset = QuickdrawDataset(dataloader_configs['test_picture_path_root'], dataloader_configs['sketch_list_4_test'], transform_val)
test_loader = DataLoader(test_dataset, batch_size=dataloader_configs['batch_size'], shuffle=False, num_workers=dataloader_configs['num_workers'])


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

# TODO
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

logger.info("basic configuration settings: {}".format(basic_configs))
#logger.info("dataloader configuration settings: {}".format(dataloader_configs))


# Part 2-4. configurations for loss function, and optimizer

# TODO
loss_function = nn.CrossEntropyLoss()

max_val_acc = 0.0
max_val_acc_epoch = -1

net = models.mobilenet_v2(num_classes = 345)
logger.info("withOUT ImageNet pretraining!!!")
net = net.cuda()
# net = torch.nn.DataParallel(net, device_ids=[int(x) for x in args.gpu.split(',')]).cuda()

# optimizer = torch.optim.SGD(net.parameters(), lr=basic_configs['learning_rate'], momentum=0.9, weight_decay=5e-4)
# TODO  change as RMSProb
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
    # TODO 
    for idx, (sketch, label) in enumerate(tqdm(train_loader, ascii=True)):

        sketch = sketch.cuda()
        label = label.cuda()
     
        optimizer.zero_grad()

        output = net(sketch)
        # ipdb.set_trace()

    
        batch_loss = loss_function(output, label)

        batch_loss.backward()

        optimizer.step()

        training_loss.update(batch_loss.item(), sketch.size(0))
        
        training_acc.update(accuracy(output, label, topk = (1,))[0].item(), sketch.size(0))

        if (idx + 1) % basic_configs["display_step"] == 0:
            logger.info(
                "==> Iteration [{}][{}/{}]:".format(epoch + 1, idx + 1, len(train_loader)))
            logger.info("current batch loss: {}".format(
                batch_loss.item()))
            logger.info("average loss: {}".format(
                training_loss.avg))
            logger.info("average acc: {}".format(training_acc.avg))  

    # TODO
    logger.info("Begin Evaluating")

    validation_loss, validation_acc = validate_function(val_loader)

    test_loss, test_acc = validate_function(test_loader)
    
    writer.add_scalars("loss", {
        "training_loss":training_loss.avg,
        "validation_loss":validation_loss.avg,
        "test_loss":test_loss.avg,
        }, epoch+1)
    writer.add_scalars("acc", {
        "training_acc":training_acc.avg,
        "validation_acc":validation_acc.avg,
        "test_acc":test_acc.avg
        }, epoch+1)

    return validation_acc



# Part 4. 'validate' function
def validate_function(data_loader):
    validation_loss = AverageMeter()
    
    validation_acc_1 = AverageMeter()
    validation_acc_5 = AverageMeter()
    validation_acc_10 = AverageMeter()
    net.eval()
    # TODO  
    with torch.no_grad():
        for idx, (sketch, label) in enumerate(tqdm(data_loader, ascii=True)):
            
            sketch = sketch.cuda()
            label = label.cuda()
        
            
            output = net(sketch)

            batch_loss = loss_function(output, label)

            validation_loss.update(batch_loss.item(), sketch.size(0))
            
            acc_1, acc_5, acc_10 = accuracy(output, label, topk = (1, 5, 10))
            validation_acc_1.update(acc_1, sketch.size(0))
            validation_acc_5.update(acc_5, sketch.size(0))
            validation_acc_10.update(acc_10, sketch.size(0))

        logger.info("==> Testing Result: ")
        logger.info("loss: {}  acc@1: {} acc@5: {} acc@10: {}".format(validation_loss.avg, validation_acc_1.avg, validation_acc_5.avg, validation_acc_10.avg))

    return validation_loss, validation_acc_1





# Part 5. 'main' function
if __name__ == '__main__':

    logger.info("Begin Evaluating before training")
    validate_function(val_loader)
    
    
    logger.info("training status: ")
    for epoch in range(basic_configs['num_epochs']):
        logger.info("Begin training epoch {}".format(epoch + 1))

        validation_acc = train_function(epoch)

        if validation_acc.avg > max_val_acc:
            max_val_acc = validation_acc.avg
            max_val_acc_epoch = epoch + 1

        logger.info("max_val_acc: {}  max_val_acc_epoch: {}".format(max_val_acc, max_val_acc_epoch))

        net_checkpoint_name = args.exp + "_net_epoch" + str(epoch + 1)
        net_checkpoint_path = os.path.join(exp_ckpt_dir, net_checkpoint_name)
        net_state = {"epoch": epoch + 1,
                     "network": net.state_dict()}
        torch.save(net_state, net_checkpoint_path)


    logger.info("max_val_acc: {}  max_val_acc_epoch: {}".format(max_val_acc, max_val_acc_epoch))



import argparse
import collections
import datetime
import os
import pickle
import time
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import random
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
from dataloader.QuickdrawDataset4dict_2nn import *
from network.gra_transf_inpt5_new_dropout_2layerMLP_3_adj_mtx import *
from utils.AverageMeter import AverageMeter
from utils.Logger import Logger
from utils.accuracy import *
from utils.EarlyStopping import *
from tqdm import tqdm



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
parser = argparse.ArgumentParser(description='MGT_stage_1')
parser.add_argument("--exp", type=str, default="train_gra_transf_inpt5_new_dropout_2layerMLP_2nn2nn2nn_early_stop_001", help="experiment")
parser.add_argument("--train_coordinate_path_root", type=str, default="/home/peng/dataset/tiny_quickdraw_coordinate/train/", help="train_sketch_coordinate_dir")
parser.add_argument("--val_coordinate_path_root", type=str, default="/home/peng/dataset/tiny_quickdraw_coordinate/val/", help="val_sketch_coordinate_dir")
parser.add_argument("--test_coordinate_path_root", type=str, default="/home/peng/dataset/tiny_quickdraw_coordinate/test/", help="test_sketch_coordinate_dir")
parser.add_argument("--sketch_list", type=str, default="./dataloader/tiny_train_set.txt", help="sketch_list_urls")
parser.add_argument("--sketch_list_4_val", type=str, default="./dataloader/tiny_val_set.txt", help="sketch_list_urls_4_validation")
parser.add_argument("--sketch_list_4_test", type=str, default="./dataloader/tiny_test_set.txt", help="sketch_list_urls_4_test")
parser.add_argument("--batch_size", type=int, default=192, help="batch_size")
parser.add_argument("--num_workers", type=int, default=12, help="num_workers")
parser.add_argument('--gpu', type=str, default="0", help='choose GPU')


args = parser.parse_args()



# Part 2. configurations

# Part 2-1. Basic configuration
basic_configs = collections.OrderedDict()
basic_configs['serial_number'] = args.exp
basic_configs['random_seed'] = int(time.time())
_seed = basic_configs['random_seed']
random.seed(_seed)
np.random.seed(_seed)
torch.manual_seed(_seed)
torch.cuda.manual_seed(_seed)
torch.cuda.manual_seed_all(_seed)
os.environ['PYTHONHASHSEED'] = str(_seed)
basic_configs['learning_rate'] = 0.00005
basic_configs['num_epochs'] = 100
basic_configs['early_stopping_patience'] = 10

basic_configs["lr_protocol"] = [(10,0.00005), (20,0.00005 * 0.7), (30,0.00005 * 0.7 * 0.7), (40,0.00005 * 0.7 * 0.7 * 0.7), (50,0.00005 * 0.7 * 0.7 * 0.7 * 0.7),    (60,0.00005 * 0.7 * 0.7 * 0.7 * 0.7 * 0.7), (70,0.00005 * 0.7 * 0.7 * 0.7 * 0.7 * 0.7 * 0.7), (80,0.00005 * 0.7 * 0.7 * 0.7 * 0.7 * 0.7 * 0.7 * 0.7), (90,0.00005 * 0.7 * 0.7 * 0.7 * 0.7 * 0.7 * 0.7 * 0.7 * 0.7), (100,0.00005 * 0.7 * 0.7 * 0.7 * 0.7 * 0.7 * 0.7 * 0.7 * 0.7 * 0.7)]


basic_configs["display_step"] = 100
lr_protocol = basic_configs["lr_protocol"]


# Part 2-2. dataloader instantiation
dataloader_configs = collections.OrderedDict()
dataloader_configs['train_coordinate_path_root'] = args.train_coordinate_path_root
dataloader_configs['val_coordinate_path_root'] = args.val_coordinate_path_root
dataloader_configs['test_coordinate_path_root'] = args.test_coordinate_path_root
dataloader_configs['sketch_list'] = args.sketch_list
dataloader_configs['sketch_list_4_val'] = args.sketch_list_4_val
dataloader_configs['sketch_list_4_test'] = args.sketch_list_4_test
dataloader_configs['batch_size'] = args.batch_size
dataloader_configs['num_workers'] = args.num_workers




dataloader_configs['data_dict_4_train'] = './dataloader/tiny_train_dataset_dict.pickle'
data_dict_4_train_f = open(dataloader_configs['data_dict_4_train'], 'rb')
data_dict_4_train = pickle.load(data_dict_4_train_f)  

dataloader_configs['data_dict_4_validation'] = './dataloader/tiny_val_dataset_dict.pickle' 
data_dict_4_validation_f = open(dataloader_configs['data_dict_4_validation'], 'rb')
data_dict_4_validation = pickle.load(data_dict_4_validation_f)  


dataloader_configs['data_dict_4_test'] = './dataloader/tiny_test_dataset_dict.pickle' 
data_dict_4_test_f = open(dataloader_configs['data_dict_4_test'], 'rb')
data_dict_4_test = pickle.load(data_dict_4_test_f)  

#ipdb.set_trace()


# create dataset
# -----------------------------------------------------------------------------------------------------

train_dataset = QuickdrawDataset_2nn(dataloader_configs['train_coordinate_path_root'], dataloader_configs['sketch_list'], data_dict_4_train)
train_loader = DataLoader(train_dataset, batch_size=dataloader_configs['batch_size'], shuffle=True, num_workers=dataloader_configs['num_workers'])

val_dataset = QuickdrawDataset_2nn(dataloader_configs['val_coordinate_path_root'], dataloader_configs['sketch_list_4_val'], data_dict_4_validation)
val_loader = DataLoader(val_dataset, batch_size=dataloader_configs['batch_size'], shuffle=False, num_workers=dataloader_configs['num_workers'])

test_dataset = QuickdrawDataset_2nn(dataloader_configs['test_coordinate_path_root'], dataloader_configs['sketch_list_4_test'], data_dict_4_test)
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


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu



logger.info("argument parser settings: {}".format(args))

logger.info("basic configuration settings: {}".format(basic_configs))



# Part 2-4. configurations for loss function, network, and optimizer


loss_function = nn.CrossEntropyLoss()


max_val_acc = 0.0
max_val_acc_epoch = -1


network_configs=collections.OrderedDict()

network_configs['output_dim']=345
network_configs['n_heads']=8
network_configs['embed_dim']=256
network_configs['n_layers']=4
network_configs['feed_forward_hidden']=4*network_configs['embed_dim']
network_configs['normalization']='batch'
network_configs['dropout']=0.25
network_configs['mlp_classifier_dropout']=0.25


logger.info("network configuration settings: {}".format(network_configs))


net = make_model(n_classes=345, coord_input_dim=2, feat_input_dim=2, feat_dict_size=103, 
                 n_layers=network_configs['n_layers'], n_heads=network_configs['n_heads'], 
                 embed_dim=network_configs['embed_dim'], feedforward_dim=network_configs['feed_forward_hidden'], 
                 normalization=network_configs['normalization'], dropout=network_configs['dropout'], mlp_classifier_dropout=network_configs['mlp_classifier_dropout'])
net = net.cuda()


 
optimizer = torch.optim.Adam(net.parameters(), lr=0.00005)


# Part 3. 'train' function
def train_function(epoch):
    training_loss = AverageMeter()
    training_acc = AverageMeter()
    net.train()

    lr = next((lr for (max_epoch, lr) in lr_protocol if max_epoch > epoch), lr_protocol[-1][1])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    logger.info("set learning rate to: {}".format(lr))

    for idx, (coordinate, label, flag_bits, stroke_len, attention_mask, padding_mask, position_encoding) in enumerate(tqdm(train_loader, ascii=True)):

        coordinate = coordinate.cuda()
        label = label.cuda()
        flag_bits = flag_bits.cuda()
        stroke_len = stroke_len.cuda()
        attention_mask = attention_mask.cuda()
    
        padding_mask = padding_mask.cuda()
        position_encoding = position_encoding.cuda()
        
        # Resize inputs
        flag_bits.squeeze_(2)
        position_encoding.squeeze_(2)
        stroke_len.unsqueeze_(1)
     
        optimizer.zero_grad()

        output = net(coordinate, flag_bits, position_encoding, attention_mask, attention_mask, attention_mask, padding_mask, stroke_len)
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

    validation_loss, validation_acc = validate_function(val_loader)

    test_loss, test_acc = validate_function(test_loader)
    
    writer.add_scalars("loss", {
        "training_loss":training_loss.avg,
        "validation_loss":validation_loss.avg,
        "test_loss":test_loss.avg
        }, epoch+1)
    writer.add_scalars("acc", {
        "training_acc":training_acc.avg,
        "validation_acc":validation_acc.avg,
        "test_acc":test_acc.avg,
        }, epoch+1)

    return validation_acc


# Part 4. 'validate' function
def validate_function(data_loader):
    validation_loss = AverageMeter()

    validation_acc_1 = AverageMeter()
    validation_acc_5 = AverageMeter()
    validation_acc_10 = AverageMeter()

    net.eval()
    
    timelist = list()
      
    with torch.no_grad():
        for idx, (coordinate, label, flag_bits, stroke_len, attention_mask, padding_mask, position_encoding) in enumerate(tqdm(data_loader, ascii=True)):
            
            coordinate = coordinate.cuda()
            label = label.cuda()
            flag_bits = flag_bits.cuda()
            stroke_len = stroke_len.cuda()
            attention_mask = attention_mask.cuda()
            
            padding_mask = padding_mask.cuda()
            position_encoding = position_encoding.cuda()

            # Resize inputs
            flag_bits.squeeze_(2)
            position_encoding.squeeze_(2)
            stroke_len.unsqueeze_(1)

             
            tic = time.time()
            
            output = net(coordinate, flag_bits, position_encoding, attention_mask, attention_mask, attention_mask, padding_mask, stroke_len)

            timelist.append(time.time() - tic)

            batch_loss = loss_function(output, label)
            

            validation_loss.update(batch_loss.item(), coordinate.size(0))
            
            
            acc_1, acc_5, acc_10 = accuracy(output, label, topk = (1, 5, 10))
            validation_acc_1.update(acc_1, coordinate.size(0))
            validation_acc_5.update(acc_5, coordinate.size(0))
            validation_acc_10.update(acc_10, coordinate.size(0))

        logger.info("==> Evaluation Result: ")
        
        logger.info("loss: {}  acc@1: {} acc@5: {} acc@10: {}".format(validation_loss.avg, validation_acc_1.avg, validation_acc_5.avg, validation_acc_10.avg))
        logger.info("Total inference time: {}s".format(sum(timelist)))

    return validation_loss, validation_acc_1




# Part 5. 'main' function
if __name__ == '__main__':

    logger.info("Begin Evaluating before training")
    validate_function(val_loader)
    
    logger.info("training status: ")

     
    early_stopping = EarlyStopping(patience=basic_configs['early_stopping_patience'], delta=0)

    for epoch in range(basic_configs['num_epochs']):
        logger.info("Begin training epoch {}".format(epoch + 1))
        validation_acc = train_function(epoch)

        if validation_acc.avg > max_val_acc:
            max_val_acc = validation_acc.avg
            max_val_acc_epoch = epoch + 1

         
        early_stopping(validation_acc.avg)
        logger.info("Early stopping counter: {}".format(early_stopping.counter))
        logger.info("Early stopping best_score: {}".format(early_stopping.best_score))
        logger.info("Early stopping early_stop: {}".format(early_stopping.early_stop))

        if early_stopping.early_stop == True:
            logger.info("Early stopping after Epoch: {}".format(epoch + 1))
            break

        
        net_checkpoint_name = args.exp + "_net_epoch" + str(epoch + 1)
        net_checkpoint_path = os.path.join(exp_ckpt_dir, net_checkpoint_name)
        net_state = {"epoch": epoch + 1,
                     "network": net.state_dict()}
        torch.save(net_state, net_checkpoint_path)

    logger.info("max_val_acc: {}  max_val_acc_epoch: {}".format(max_val_acc, max_val_acc_epoch))    

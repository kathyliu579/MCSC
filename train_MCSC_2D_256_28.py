# -*- coding: utf-8 -*-
# Author: Qianying liu
# Date:   May. 2023
# Implementation for Multi-Scale Cross Contrastive Learning for Semi-Supervised Medical Image Segmentation (BMVC 2023).
# # Reference:
#   @article{luo2021ctbct,
#   title={Semi-Supervised Medical Image Segmentation via Cross Teaching between CNN and Transformer},
#   author={Luo, Xiangde and Hu, Minhao and Song, Tao and Wang, Guotai and Zhang, Shaoting},
#   journal={arXiv preprint arXiv:2112.04894},
#   year={2021}}


import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from config import get_config
from dataloaders import utils
from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler)
from networks.net_factory import net_factory
from networks.vision_transformer import SwinUnet as ViT_seg
from utils import losses, metrics, ramps
from val_2D import test_single_volume
from utils.supcon_loss_romdom_seleted import BlockConLoss

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Cross_Teaching_Between_CNN_Transformer', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=40000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=5,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=4,
                    help='output channel of network')
parser.add_argument(
    '--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=4,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=6,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()
config = get_config(args)


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=15,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        warmup_epochs = int(epochs / 20)
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim, kernel):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=kernel, padding=1)

        #  init my layers
        nn.init.xavier_normal_(self.conv1.weight)

    def forward(self, x, dropout=True):
        x = self.conv1(x)

        return x


class Projector(nn.Module):
    def __init__(self, in_dim1, in_dim2, out_dim, downsample=False):
        super(Projector, self).__init__()

        self.in_dim1 = in_dim1  # 16
        self.in_dim2 = in_dim2  # 96
        self.out_dim = out_dim  # 128
        self.downsample = downsample
        self.conv1 = nn.Conv2d(self.in_dim1, 256, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.in_dim2, 256, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(256, self.out_dim, kernel_size=1, stride=1)

    def forward(self, x1, x2):
        if self.downsample:
            x1 = F.avg_pool2d(x1, kernel_size=2, stride=2)
            x2 = F.avg_pool2d(x2, kernel_size=2, stride=2)
        x1 = self.conv3(self.relu(self.conv1(x1)))
        x2 = self.conv3(self.relu(self.conv2(x2)))
        return x1, x2
    
    
class Projector3(nn.Module):
    def __init__(self, in_dim1, in_dim2, out_dim):
        super(Projector3, self).__init__()

        self.in_dim1 = in_dim1  # 5, 128, 28, 28]   
        self.in_dim2 = in_dim2  #  [5, 784, 192] = [5,  192 , 28, 28] 
        self.out_dim = out_dim  # 128
        self.conv1 = nn.Conv2d(self.in_dim1, 256, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.LayerNorm(self.in_dim2)
        self.conv2 = nn.Conv2d(self.in_dim2, 256, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(256, self.out_dim, kernel_size=1, stride=1)

    def forward(self, x1, x2):
        x1 = self.conv3(self.relu(self.conv1(x1)))
        
        # process x2 into conv shape 
        x2 =  self.norm(x2)
        x2 = x2.view(-1, 28, 28, self.in_dim2)
        x2 = x2.permute(0, 3, 1, 2)
        x2 = self.conv3(self.relu(self.conv2(x2)))
        return x1, x2


########## revise projector and classifier parameter ////   delete unet classifier


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 23,"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model1 = create_model().cuda()
    model2 = ViT_seg(config, img_size=args.patch_size,
                     num_classes=args.num_classes).cuda()
    model2.load_from(config)

    projector = Projector(in_dim1=16, in_dim2=96, out_dim=128).cuda()
    projector3 = Projector3(in_dim1=128, in_dim2=192, out_dim=128).cuda()
    classifier1 = Classifier(in_dim=16, out_dim=4, kernel=3).cuda()
    classifier2 = Classifier(in_dim=96, out_dim=4, kernel=3).cuda()

    classifier1.train()
    projector.train()
    projector3.train()
    classifier2.train()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)]))

    db_val = BaseDataSets(base_dir=args.root_path, split="val")


    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))

    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model1.train()
    model2.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    para1 = list(model1.parameters()) + list(classifier1.parameters())
    para2 = list(model2.parameters()) + list(classifier2.parameters())
    para3 = list(projector.parameters()) + list(projector3.parameters())

    proj_lr = 5e-4

    optimizer1 = optim.AdamW(para1, lr=base_lr, weight_decay=0.0005)
    optimizer2 = optim.AdamW(para2, lr=base_lr/5, weight_decay=0.0005)
    optimizer3 = optim.AdamW(para3, lr=proj_lr, weight_decay=0.0005)

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1

    lr_scheduler = create_lr_scheduler(optimizer2, len(trainloader), max_epoch, warmup=True)

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    criterion = BlockConLoss(temperature=0.1, block_size=19)    #  256/16 =16 256/19 =13  
    criterion3 = BlockConLoss(temperature=0.1, block_size=14)   #  28/14 =4

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    best_performance1 = 0.0
    best_performance2 = 0.0

    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs1_all = model1(volume_batch)
            outputs1 = outputs1_all[3]         
            outputs11 = classifier1(outputs1)
            outputs_soft1 = torch.softmax(outputs11, dim=1)
            
            outputs2_all = model2(volume_batch)
            outputs2 = outputs2_all[3]
            outputs22 = classifier2(outputs2) 
            outputs_soft2 = torch.softmax(outputs22, dim=1)  

            consistency_weight = get_current_consistency_weight(
                iter_num // 150)

            loss1 = 0.5 * (ce_loss(outputs11[:args.labeled_bs], label_batch[:args.labeled_bs].long()) + dice_loss(
                outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            loss2 = 0.5 * (ce_loss(outputs22[:args.labeled_bs], label_batch[:args.labeled_bs].long()) + dice_loss(
                outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            
            # ----------------------------------------------------------------- unsupervised----------------------------------------------------------------
            
            pseudo_outputs1 = torch.argmax(
                outputs_soft1[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs2 = torch.argmax(
                outputs_soft2[args.labeled_bs:].detach(), dim=1, keepdim=False)

            pseudo_supervision1 = dice_loss(
                outputs_soft1[args.labeled_bs:], pseudo_outputs2.unsqueeze(1))
            pseudo_supervision2 = dice_loss(
                outputs_soft2[args.labeled_bs:], pseudo_outputs1.unsqueeze(1))

            ####-------------------------------------- contrastive learning /// update (unsupervised all data)
            ####-------------for final layer 

            outputs_proj1, outputs_proj2 = projector(outputs1, outputs2)  

            label_batch2 = torch.cat(
                [label_batch[:args.labeled_bs].unsqueeze(1), label_batch[:args.labeled_bs].unsqueeze(1)], dim=1)

            corss_label = torch.cat([pseudo_outputs2.unsqueeze(1), pseudo_outputs1.unsqueeze(1)], dim=1)
            labels = torch.cat([label_batch2, corss_label], dim=0)  

            features = torch.cat([outputs_proj1, outputs_proj2], dim=0) 
            features = F.normalize(features, p=2, dim=1)
            f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1) 
            features = features.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            
            if iter_num >= 3000: 
                supconloss = criterion(features, labels)  
                if torch.isnan(supconloss):
                    raise
            else:
                supconloss = criterion(features[:args.labeled_bs], labels[:args.labeled_bs])
                
                
            ####-------------for third to last layer                 
            outputs1_l1 = outputs1_all[0]   
            outputs2_l2 = outputs2_all[1]           
            outputs_proj1, outputs_proj2 = projector3(outputs1_l1, outputs2_l2) 
            
            features = torch.cat([outputs_proj1, outputs_proj2], dim=0) 
            features = F.normalize(features, p=2, dim=1)
            f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1) 
           
            from scipy.ndimage import zoom
            
            labels = zoom(labels.cpu().detach().numpy(), (1, 1, 28 / 224, 28 / 224 ), order=0)
            labels = torch.from_numpy(labels).long()
            
            features = features.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            
            if iter_num >= 3000: 
                supconloss3 = criterion3(features, labels) 
            else:
                supconloss3 = criterion3(features[:args.labeled_bs], labels[:args.labeled_bs])
         
         
            model1_loss = loss1 + consistency_weight * pseudo_supervision1 + 0.001 * (supconloss+supconloss3)
            model2_loss = loss2 + consistency_weight * pseudo_supervision2 + 0.001 * (supconloss+supconloss3)

            if iter_num % 200 == 0:
                print("sup_loss: {:.3f}, cross_teach:{:.5f}, supconloss:{:.5f}, supconloss3:{:.5f}".format(loss1.item(),
                                                                                       consistency_weight * pseudo_supervision1.item(),
                                                                                       0.001 * supconloss.item(),0.001 * supconloss3.item()))
                
            loss = model1_loss + model2_loss
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()
            optimizer3.step()

            lr_scheduler.step()
            lr_tansformer = optimizer2.param_groups[0]["lr"]

            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            proj_lr_ = proj_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_

            for param_group in optimizer3.param_groups:
                param_group['lr'] = proj_lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('lr', lr_tansformer, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('train_loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('train_loss/model1_supervised_loss',
                              loss1, iter_num)
            writer.add_scalar('train_loss/model1_crossteach_loss',
                              pseudo_supervision1, iter_num)
            writer.add_scalar('train_loss/contrast_loss',
                              supconloss, iter_num)
            writer.add_scalar('train_loss/model2_loss',
                              model2_loss, iter_num)
            writer.add_scalar('train_loss/model2_supervised_loss',
                              loss2, iter_num)
            writer.add_scalar('train_loss/model2_crossteach_loss',
                              pseudo_supervision2, iter_num)

            # ----------------------------------------------------------------save prediction results for visualization  ---------------------------------------------------------------
            if iter_num % 200 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs11, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model1_Prediction',
                                 outputs[1, ...] * 50, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs22, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model2_Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            # ----------------------------------------------------------------model 1 : validation # validation# validation# validation----------------------------------------------------------------
            if iter_num > 1000 and iter_num % 200 == 0:
                model1.eval()
                classifier1.eval()
            
                # model 1 : validation # validation# validation# validation
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model1, classifier1, classes=num_classes,
                        patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance1 = np.mean(metric_list, axis=0)[0]
                mean_hd951 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model1_val_mean_dice',
                                  performance1, iter_num)
                writer.add_scalar('info/model1_val_mean_hd95',
                                  mean_hd951, iter_num)

                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model1_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance1, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model1.pth'.format(args.model))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)

                    save_mode_path = os.path.join(snapshot_path,
                                                  'classifier1_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance1, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_classifier1.pth'.format(args.model))
                    torch.save(classifier1.state_dict(), save_mode_path)
                    torch.save(classifier1.state_dict(), save_best)

                logging.info(
                    'iteration %d : vali_model1_mean_dice : %f model1_mean_hd95 : %f' % (
                        iter_num, performance1, mean_hd951))

                # ---------------------------------------------------------------- model 2 : validation # validation# validation# validation----------------------------------------------------------------

                model2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model2, classifier2, classes=num_classes,
                        patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/model2_val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model2_val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance2 = np.mean(metric_list, axis=0)[0]
                mean_hd952 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model2_val_mean_dice',
                                  performance2, iter_num)
                writer.add_scalar('info/model2_val_mean_hd95',
                                  mean_hd952, iter_num)

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model2_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance2, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model2.pth'.format(args.model))
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best)

                    save_mode_path = os.path.join(snapshot_path,
                                                  'classifier2_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance2, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_classifier2.pth'.format(args.model))
                    torch.save(classifier2.state_dict(), save_mode_path)
                    torch.save(classifier2.state_dict(), save_best)

                logging.info(
                    'iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance2, mean_hd952))
                model2.train()
                classifier2.train()

            if iter_num % 4000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'model1_iter_{}_dice_{}.pth'.format(
                        iter_num, round(performance1, 4)))
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))
                save_mode_path = os.path.join(snapshot_path,
                                              'classifier1_iter_{}_dice_{}.pth'.format(
                                                  iter_num, round(performance1, 4)))
                torch.save(classifier1.state_dict(), save_mode_path)

                save_mode_path = os.path.join(
                    snapshot_path, 'model2_iter_{}_dice_{}.pth'.format(
                        iter_num, round(performance2, 4)))
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))
                save_mode_path = os.path.join(snapshot_path,
                                              'classifier2_iter_{}_dice_{}.pth'.format(
                                                  iter_num, round(performance2, 4)))
                torch.save(classifier2.state_dict(), save_mode_path)

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    print('consistency_weight', consistency_weight)
    writer.close()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
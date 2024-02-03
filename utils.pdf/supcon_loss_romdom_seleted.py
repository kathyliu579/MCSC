from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from util import get_gpu_memory_map
import numpy


class SupConLoss(nn.Module):
    """modified supcon loss for segmentation application, also it is a balanced loss, which averages the contributions of each category in the denominator """

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None):
        # input features shape: [bsz, v, c, w, h]
        # input labels shape: [bsz, v, w, h]
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        contrast_count = 2
        contrast_feature = features

        kernels = contrast_feature.permute(0, 2, 3, 1)
        kernels = kernels.reshape(-1, contrast_feature.shape[1], 1, 1)
        logits = torch.div(F.conv2d(contrast_feature, kernels), self.temperature)  # of size (bsz*v, bsz*v*h*w, h, w)
        logits = logits.permute(1, 0, 2, 3)
        logits = logits.reshape(logits.shape[0], -1)

        if labels is not None:
            labels = torch.cat(torch.unbind(labels, dim=1), dim=0)
            labels = labels.contiguous().view(-1, 1)
            batch_cls_count = torch.eye(4)[labels].sum(dim=0).squeeze()
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = torch.eye(logits.shape[0] // contrast_count).float().to(device)
            mask = mask.repeat(contrast_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(mask.shape[0]).view(-1, 1).to(device),
            0)
        mask = mask * logits_mask
        # compute balanced log_prob
        exp_logits = torch.exp(logits) * logits_mask
        per_ins_weight = torch.tensor([batch_cls_count[i] for i in labels.squeeze()], device=device).view(1, -1) - mask
        exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)
        log_prob = logits - torch.log(exp_logits_sum + 1e-6)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

        # loss
        loss = - mean_log_prob_pos
        loss = loss.mean()
        return loss


class BlockConLoss(nn.Module):
    def __init__(self, temperature=0.7, block_size=32):
        super(BlockConLoss, self).__init__()
        self.block_size = block_size
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.supconloss = SupConLoss(temperature=temperature)
        self.stride = self.block_size // 2

    def forward(self, features, labels=None):
        # input features: [bsz, num_view, c, h ,w], h & w are the image size;  labels: [bsz, num_view, h ,w]
        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        labels = torch.cat(torch.unbind(labels, dim=1), dim=0)
        shape = features.shape
        img_size = shape[-1]
        div_num = img_size // (self.block_size ) - 1  # 10    

        if labels is not None:
            loss = []
            features_list = []
            label_list = []
            # start.record()
            for i in range(div_num):
                # print("Iteration index:", idx, "Batch_size:", b)
                for j in range(div_num):
                    block_features = features[:,  :, i*self.block_size:(i+1)*self.block_size,
                                  j*self.block_size:(j+1)*self.block_size]                   
                    block_labels = labels[:, i*self.block_size:(i+1)*self.block_size,
                                  j*self.block_size:(j+1)*self.block_size]
                    features_list.append(block_features)
                    label_list.append(block_labels)

            features_list = torch.stack(features_list, dim=0)
            label_list = torch.stack(label_list, dim=0)
            
            features_list = list(torch.unbind(features_list, dim= 1)) #  8* [100, 128, 19, 19
            label_list = list(torch.unbind(label_list, dim= 1))
           
            for i in range(shape[0]):
                shuffle = random.sample(range(0, div_num * div_num), div_num * div_num)    
                features_list[i] = features_list[i][shuffle]
                label_list[i] = label_list[i][shuffle]
                   
            features_list = torch.stack(features_list, dim=0) #[8, 100, 128, 19, 19]
            label_list = torch.stack(label_list, dim=0) #[8, 100, 19, 19]

            for k in range(div_num * div_num):             
                tmp_loss = self.supconloss(features_list[:,k,:,:,:], label_list[:,k,:,:])
                if label_list[:,k,:,:].sum() == 0:                
                    continue
                loss.append(tmp_loss)
            if len(loss) == 0:
                loss = torch.tensor(0).float().to(self.device)
                return loss
            loss = torch.stack(loss).mean()
            return loss

        else:
            loss = []
            for i in range(div_num):
                for j in range(div_num):
                    block_features = features[:, :, :, i * self.block_size:(i + 1) * self.block_size,
                                     j * self.block_size:(j + 1) * self.block_size]
                    tmp_loss = self.supconloss(block_features)
                    loss.append(tmp_loss)
            loss = torch.stack(loss).mean()
            return loss
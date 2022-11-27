import torchvision
import pytorch_lightning as pl
import lightly

import argparse
import logging
import math
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import glob

import torch.nn as nn

from collections import OrderedDict
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from torchvision import transforms, datasets


def get_cifar10_test():

    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])

    test_dataset = datasets.CIFAR10(
        "./data", train=False, transform=transform_val, download=False)

    test_loader = DataLoader(
        test_dataset, shuffle=False, batch_size=128, num_workers=4)

    return test_loader


def get_outputs(dataloader, net):

    def euclid_dist(proto, rep):
        n = rep.shape[0]
        k = proto.shape[0]
        rep = rep.unsqueeze(1).expand(n, k, -1)
        proto = proto.unsqueeze(0).expand(n, k, -1)
        logits = -((rep - proto)**2).sum(dim=2)
        return logits

    all_out = []
    loss, acc, count = 0.0, 0.0, 0.0
    net.eval()

    prototypes = net.prototypes
    all_out = []
    with torch.no_grad():
        for dat, labels in dataloader:
            dat = dat.cuda()

            batch_size = int(labels.size()[0])
            rep = net(dat)
            if (~(rep.isfinite())).any():
                dat = dat
                rep = net(dat)
                rep = torch.nan_to_num(rep, nan=0.0, posinf=0, neginf=0) 

            out = euclid_dist(prototypes, rep)
            out = out.cpu().detach()
            all_out.append(torch.nn.functional.softmax(out, dim=1))

            out = out.numpy()
            acc += np.sum(labels.numpy() == (np.argmax(out, axis=1)))
            count += batch_size

    acc = acc / count
    all_out = np.concatenate(all_out)

    return all_out, acc


def test_acc(dataloader, net):

    all_out = []
    loss, acc, count = 0.0, 0.0, 0.0
    net.eval()

    all_out = []
    with torch.no_grad():
        for dat, labels in dataloader:
            dat = dat.cuda()

            batch_size = int(labels.size()[0])
            out = net(dat)
            out = out.cpu().detach()
            all_out.append(torch.nn.functional.softmax(out, dim=1))

            out = out.numpy()
            acc += np.sum(labels.numpy() == (np.argmax(out, axis=1)))
            count += batch_size

    acc = acc / count
    all_out = np.concatenate(all_out)

    return all_out, acc


class BartonTwins(nn.Module):
    def __init__(self):
        super().__init__()
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18')
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )
        # create a simsiam model based on ResNet
        # note that bartontwins has the same architecture
        self.resnet_simsiam = \
            lightly.models.SimSiam(self.backbone, num_ftrs=512, num_mlp_layers=3)
            
    def forward(self, x):
        out = self.backbone(x)
        out = out.reshape(-1, 512)
        return out


def get_prototype(net, dataloader):

    prototypes = torch.zeros(10, 512)
    lab_count = torch.zeros(10)

    prototypes = prototypes.cuda(0)
    lab_count = lab_count.cuda(0)
    num_cls = 10

    net.eval()
    with torch.no_grad():
        for idx, (dat, labels) in enumerate(dataloader):
            dat = dat.cuda()
            labels = labels.cuda()
            rep = net(dat)

            if (~(rep.isfinite())).any():
                dat = dat
                rep = net(dat)
                rep = torch.nan_to_num(rep, nan=0.0, posinf=0, neginf=0) 

            prototypes.index_add_(0, labels, rep)
            lab_count += torch.bincount(labels, minlength=num_cls)

    prototypes = prototypes / lab_count.reshape(-1, 1)
    net.prototypes = prototypes

    return prototypes


def get_predictions(fdir):

    ckpt_list = []

    epoch_count = np.zeros(1000)
    for fname in glob.glob(fdir + "/*.ckpt"):
        steps = int((fname.split(".")[0]).split("=")[-1])
        epoch = int((fname.split("-")[0]).split("=")[-1])
        if epoch >= 1:
            if epoch_count[epoch] >= 1:
                continue
            epoch_count[epoch] += 1
        ckpt_list.append((steps, epoch, fname))

    ckpt_list = sorted(ckpt_list)
    testloader = get_cifar10_test()

    allpreds = []
    count = 0
    for ckpt_name in ckpt_list:
        count += 1
        print(ckpt_name)
        net = BartonTwins()

        ckpt = torch.load(ckpt_name[-1])
        net.load_state_dict(ckpt['state_dict'])
        net.cuda()

        prototypes = get_prototype(net, testloader)
        preds, acc = get_outputs(testloader, net)
        allpreds.append(preds)

        print("Acc: %.2f" % acc)

    allpreds = np.array(allpreds)
    npy_name = "probs1" + fdir.split("1")[-1]
    np.save("./ckpt/predictions/" + npy_name, allpreds)


# get_predictions("ckpt/barlow1")
# get_predictions("ckpt/barlow10")
get_predictions("ckpt/barlow100")
get_predictions("ckpt/barlow1000")
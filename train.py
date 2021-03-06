"""
created by: Donghyeon Won
Modified codes from
    http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    https://github.com/pytorch/Fexamples/tree/master/imagenet
"""

from __future__ import print_function
import os
import argparse
import numpy as np
import pandas as pd
import time
import shutil
# from itertools import ifilter
# from future_builtins import map, filter
from PIL import Image
from sklearn.metrics import accuracy_score, mean_squared_error

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models

# from util import ProtestDataset, modified_resnet50, modified_resnet18,AverageMeter, Lighting, *
from util import  *


# for indexing output of the model
protest_idx = Variable(torch.LongTensor([0]))
violence_idx = Variable(torch.LongTensor([1]))
visattr_idx = Variable(torch.LongTensor(range(2,12)))
best_loss = float("inf")

def calculate_loss(output, target, criterions, weights = [1, 10, 5]):
    """Calculate loss"""
    # number of protest images
    N_protest = int(target['protest'].data.sum())
    batch_size = len(target['protest'])

    if N_protest == 0:
        # if no protest image in target
        outputs = [None]
        # protest output
        outputs[0] = output.index_select(1, protest_idx)
        targets = [None]
        # protest target
        targets[0] = target['protest'].float()
        losses = [weights[i] * criterions[i](outputs[i], targets[i]) for i in range(1)]
        scores = {}
        # output = output.cpu().data
        # targets = targets.data
        # scores['protest_acc'] = accuracy_score((outputs[0]).data.round(), targets[0].data)
        scores['protest_acc'] = accuracy_score((outputs[0]).cpu().data.round(), targets[0].cpu().data)
        scores['violence_mse'] = 0
        scores['visattr_acc'] = 0
        return losses, scores, N_protest

    # used for filling 0 for non-protest images
    not_protest_mask = (1 - target['protest']).byte()
    # print(type(output))
    outputs = [None] * 4
    # protest output
    outputs[0] = output.index_select(1, protest_idx)
    # violence output
    outputs[1] = output.index_select(1, violence_idx)
    # outputs[1].masked_fill_(not_protest_mask, 0)
    outputs[1].bool().masked_fill_(not_protest_mask, 0)
    # visual attribute output
    outputs[2] = output.index_select(1, visattr_idx)
    # outputs[2].masked_fill_(not_protest_mask.repeat(1, 10),0)
    outputs[2].bool().masked_fill_(not_protest_mask.repeat(1, 10),0)


    targets = [None] * 4

    targets[0] = target['protest'].float()
    targets[1] = target['violence'].float()
    targets[2] = target['visattr'].float()

    scores = {}
    # protest accuracy for this batch
    scores['protest_acc'] = accuracy_score(outputs[0].cpu().data.round(), targets[0].cpu().data)
    # violence MSE for this batch
    scores['violence_mse'] = ((outputs[1].cpu().data - targets[1].cpu().data).pow(2)).sum() / float(N_protest)
    # mean accuracy for visual attribute for this batch
    comparison = (outputs[2].cpu().data.round() == targets[2].cpu().data)
    comparison.masked_fill_(not_protest_mask.repeat(1, 10).cpu().data,0)
    n_right = comparison.float().sum()
    mean_acc = n_right / float(N_protest*10)
    scores['visattr_acc'] = mean_acc

    # return weighted loss
    losses = [weights[i] * criterions[i](outputs[i], targets[i]) for i in range(len(criterions))]

    return losses, scores, N_protest



def train(train_loader, model, criterions, optimizer, epoch):
    """training the model"""

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_protest = AverageMeter()
    loss_v = AverageMeter()
    protest_acc = AverageMeter()
    violence_mse = AverageMeter()
    visattr_acc = AverageMeter()

    end = time.time()
    loss_history = []
    for i, sample in enumerate(train_loader):
        # measure data loading batch_time
        input, target = sample['image'], sample['label']
        # print(input.size())
        # print(target)
        data_time.update(time.time() - end)

        if args.cuda:
            input = input.cuda()
            for k, v in target.items():
                target[k] = v.cuda()
        target_var = {}

        for k,v in target.items():
            target_var[k] = Variable(v)

        input_var = Variable(input)
        # print(input_var.shape)
        # print("input_var",input_var.shape)
        output = model(input_var)
        # print(output.shape)
        # print("output",type(output))

        losses, scores, N_protest = calculate_loss(output, target_var, criterions)

        optimizer.zero_grad()
        loss = 0
        for l in losses:
            loss += l
        # back prop
        loss.backward()
        optimizer.step()
        # print("here",loss,losses)

        if N_protest:
            # loss_protest.update(losses[0].data[0], input.size(0))
            # print("1",losses[0].item())
            loss_protest.update(losses[0].item(), input.size(0))

            # loss_v.update(loss.data[0] - losses[0].data[0], N_protest)
            loss_v.update(loss.item() - losses[0].item(), N_protest)
        else:
            # when there is no protest image in the batch
            # print(losses)
            # print(losses[0])
            # loss_protest.update(losses[0].data[0].item(), input.size(0))

            loss_protest.update(losses[0].item(), input.size(0))

        # print("2",loss)

        # loss_history.append(loss.data[0])
        loss_history.append(loss.item())
        protest_acc.update(scores['protest_acc'], input.size(0))
        violence_mse.update(scores['violence_mse'], N_protest)
        visattr_acc.update(scores['visattr_acc'], N_protest)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}] '
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})  '
                  'Data {data_time.val:.2f} ({data_time.avg:.2f})  '
                  'Loss {loss_val:.3f} ({loss_avg:.3f})  '
                  'Protest {protest_acc.val:.3f} ({protest_acc.avg:.3f})  '
                  'Violence {violence_mse.val:.5f} ({violence_mse.avg:.5f})  '
                  'Vis Attr {visattr_acc.val:.3f} ({visattr_acc.avg:.3f})'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time,
                   loss_val=loss_protest.val + loss_v.val,
                   loss_avg = loss_protest.avg + loss_v.avg,
                   protest_acc = protest_acc, violence_mse = violence_mse,
                   visattr_acc = visattr_acc))

    return loss_history

def validate(val_loader, model, criterions, epoch):
    """Validating"""
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_protest = AverageMeter()
    loss_v = AverageMeter()
    protest_acc = AverageMeter()
    violence_mse = AverageMeter()
    visattr_acc = AverageMeter()

    end = time.time()
    loss_history = []
    for i, sample in enumerate(val_loader):
        # measure data loading batch_time
        input, target = sample['image'], sample['label']

        if args.cuda:
            input = input.cuda()
            for k, v in target.items():
                target[k] = v.cuda()
        input_var = Variable(input)

        target_var = {}
        for k,v in target.items():
            target_var[k] = Variable(v)

        output = model(input_var)

        losses, scores, N_protest = calculate_loss(output, target_var, criterions)
        loss = 0
        for l in losses:
            loss += l

        if N_protest:
            # loss_protest.update(losses[0].data[0], input.size(0))
            # loss_v.update(loss.data[0] - losses[0].data[0], N_protest)
            loss_protest.update(losses[0].data, input.size(0))
            loss_v.update(loss.data - losses[0].data, N_protest)
        else:
            # when no protest images
            # loss_protest.update(losses[0].data[0], input.size(0))
            loss_protest.update(losses[0].data, input.size(0))
        loss_history.append(loss.data)
        protest_acc.update(scores['protest_acc'], input.size(0))
        violence_mse.update(scores['violence_mse'], N_protest)
        visattr_acc.update(scores['visattr_acc'], N_protest)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})  '
                  'Loss {loss_val:.3f} ({loss_avg:.3f})  '
                  'Protest Acc {protest_acc.val:.3f} ({protest_acc.avg:.3f})  '
                  'Violence MSE {violence_mse.val:.5f} ({violence_mse.avg:.5f})  '
                  'Vis Attr Acc {visattr_acc.val:.3f} ({visattr_acc.avg:.3f})'
                  .format(
                   epoch, i, len(val_loader), batch_time=batch_time,
                   loss_val =loss_protest.val + loss_v.val,
                   loss_avg = loss_protest.avg + loss_v.avg,
                   protest_acc = protest_acc,
                   violence_mse = violence_mse, visattr_acc = visattr_acc))

    print(' * Loss {loss_avg:.3f} Protest Acc {protest_acc.avg:.3f} '
          'Violence MSE {violence_mse.avg:.5f} '
          'Vis Attr Acc {visattr_acc.avg:.3f} '
          .format(loss_avg = loss_protest.avg + loss_v.avg,
                  protest_acc = protest_acc,
                  violence_mse = violence_mse, visattr_acc = visattr_acc))
    return loss_protest.avg + loss_v.avg, loss_history

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 0.5 every 5 epochs"""
    lr = args.lr * (0.4 ** (epoch // 4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Save checkpoints"""
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def main():
    global best_loss
    loss_history_train = []
    loss_history_val = []
    data_dir = args.data_dir
    img_dir_train = os.path.join(data_dir, "img/train")
    img_dir_val = os.path.join(data_dir, "img/test")
    # txt_file_train = os.path.join(data_dir, "annot_train.txt")
    txt_file_train = os.path.join(data_dir, "train.txt")
    # txt_file_val = os.path.join(data_dir, "annot_test.txt")
    txt_file_val = os.path.join(data_dir, "test.txt")

    # load pretrained resnet50 with a modified last fully connected layer
    if args.model == "resnet50":
        model = modified_resnet50()
        print("model ------------>",args.model,model)
    elif args.model == "resnet18":
        model = modified_resnet18()
        print("model ------------>",args.model,model)
    elif args.model == "resnet34":
        model = modified_resnet18()
        print("model ------------>",args.model,model)
    elif args.model == "inceptionv3":
        model = modified_inception()
        print("model ------------>",args.model,model)
    elif args.model == "squeeznet":
        model = modified_squeeznet()
        print("model ------------>",args.model,model)
    elif args.model == "densenet":
        # model = DenseNet(32, (6, 12, 32, 32),num_classes=12)
        model = modified_densenet161()
        print("model ------------>",args.model,model)
    elif args.model == "eff_resnet":
        model = modified_eff_resnet()
        print("model ------------>",args.model,model)
    elif args.model == "single_perceptron":
        model = single_perceptron()
        print("model ------------>", args.model,model)

    # we need three different criterion for training
    criterion_protest = nn.BCELoss()
    criterion_violence = nn.MSELoss()
    criterion_visattr = nn.BCELoss()
    criterions = [criterion_protest, criterion_violence, criterion_visattr]

    if args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU Found")
    if args.cuda:
        model = model.cuda()
        criterions = [criterion.cuda() for criterion in criterions]
    # we are not training the frozen layers
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.SGD(
                        parameters, args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay
                        )

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            loss_history_train = checkpoint['loss_history_train']
            loss_history_val = checkpoint['loss_history_val']
            if args.change_lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr
            else:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    eigval = torch.Tensor([0.2175, 0.0188, 0.0045])
    eigvec = torch.Tensor([[-0.5675,  0.7192,  0.4009],
                           [-0.5808, -0.0045, -0.8140],
                           [-0.5836, -0.6948,  0.4203]])

    
    if args.model == "inceptionv3":
        train_dataset = ProtestDataset(
                    txt_file = txt_file_train,
                    img_dir = img_dir_train,
                    transform = transforms.Compose([
                            transforms.RandomResizedCrop(299),
                            transforms.RandomRotation(30),
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(
                                brightness = 0.4,
                                contrast = 0.4,
                                saturation = 0.4,
                                ),
                            transforms.ToTensor(),
                            Lighting(0.1, eigval, eigvec),
                            normalize,
                    ]))
        val_dataset = ProtestDataset(
                        txt_file = txt_file_val,
                        img_dir = img_dir_val,
                        transform = transforms.Compose([
                            transforms.Resize(350),
                            transforms.CenterCrop(299),
                            transforms.ToTensor(),
                            normalize,
                        ]))
    else:



        train_dataset = ProtestDataset(
                            txt_file = txt_file_train,
                            img_dir = img_dir_train,
                            transform = transforms.Compose([
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomRotation(30),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ColorJitter(
                                        brightness = 0.4,
                                        contrast = 0.4,
                                        saturation = 0.4,
                                        ),
                                    transforms.ToTensor(),
                                    Lighting(0.1, eigval, eigvec),
                                    normalize,
                            ]))
        val_dataset = ProtestDataset(
                        txt_file = txt_file_val,
                        img_dir = img_dir_val,
                        transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
                        ]))
    train_loader = DataLoader(
                    train_dataset,
                    num_workers = args.workers,
                    batch_size = args.batch_size,
                    shuffle = True
                    )
    val_loader = DataLoader(
                    val_dataset,
                    num_workers = args.workers,
                    batch_size = args.batch_size)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        loss_history_train_this = train(train_loader, model, criterions,
                                        optimizer, epoch)
        loss_val, loss_history_val_this = validate(val_loader, model,
                                                   criterions, epoch)
        loss_history_train.append(loss_history_train_this)
        loss_history_val.append(loss_history_val_this)

        # loss = loss_val.avg

        is_best = loss_val < best_loss
        if is_best:
            print('best model!!')
            torch.save(model, str(args.model)+"_best"+".pth")
        best_loss = min(loss_val, best_loss)
        torch.save(model, str(args.model)+"_last"+".pth")

        save_checkpoint({
            'epoch' : epoch + 1,
            'state_dict' : model.state_dict(),
            'best_loss' : best_loss,
            'optimizer' : optimizer.state_dict(),
            'loss_history_train': loss_history_train,
            'loss_history_val': loss_history_val
        }, is_best)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        type=str,
                        default = "UCLA-protest",
                        help = "directory path to UCLA-protest",
                        )
    parser.add_argument("--cuda",
                        action = "store_true",
                        help = "use cuda?",
                        )
    parser.add_argument("--workers",
                        type = int,
                        default = 0,
                        help = "number of workers",
                        )
    parser.add_argument("--batch_size",
                        type = int,
                        default = 90 ,
                        help = "batch size",
                        )
    parser.add_argument("--epochs",
                        type = int,
                        default = 40,
                        help = "number of epochs",
                        )
    parser.add_argument("--weight_decay",
                        type = float,
                        default = 1e-4,
                        help = "weight decay",
                        )
    parser.add_argument("--lr",
                        type = float,
                        default = 0.01,
                        help = "learning rate",
                        )
    parser.add_argument("--momentum",
                        type = float,
                        default = 0.9,
                        help = "momentum",
                        )
    parser.add_argument("--print_freq",
                        type = int,
                        default = 10,
                        help = "print frequency",
                        )
    parser.add_argument('--resume',
                        default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--change_lr',
                        action = "store_true",
                        help = "Use this if you want to \
                        change learning rate when resuming")
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

    parser.add_argument('--model', default="single_perceptron", type=str, metavar='N',
                    help='manual select a model in "resnet18,resnet34, resnet50, *squeeznet,*inceptionv3, densenet, eff_resnet, single_perceptron "') # for the others will release 
    args = parser.parse_args()
    print(args.cuda)
    args.cuda = True
    print(str(args.model))
    if args.cuda:
        print("here")
        protest_idx = protest_idx.cuda()
        violence_idx = violence_idx.cuda()
        visattr_idx = visattr_idx.cuda()


    main()

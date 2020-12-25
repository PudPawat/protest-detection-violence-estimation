"""
created by: Donghyeon Won
"""

from __future__ import print_function
import os
import argparse
import numpy as np
import pandas as pd
import time
import shutil
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision.models as models

from util import *


def eval_one_dir(img_dir, model):
        """
        return model output of all the images in a directory
        """
        model.eval()
        # make dataloader
        dataset = ProtestDatasetEval(img_dir = img_dir)
        data_loader = DataLoader(dataset,
                                num_workers = args.workers,
                                batch_size = args.batch_size)
        # load model

        outputs = []
        imgpaths = []

        n_imgs = len(os.listdir(img_dir))
        with tqdm(total=n_imgs) as pbar:
            for i, sample in enumerate(data_loader):
                imgpath, input = sample['imgpath'], sample['image']
                if args.cuda:
                    input = input.cuda()

                input_var = Variable(input)
                output = model(input_var)
                outputs.append(output.cpu().data.numpy())
                imgpaths += imgpath
                if i < n_imgs / args.batch_size:
                    pbar.update(args.batch_size)
                else:
                    pbar.update(n_imgs%args.batch_size)


        df = pd.DataFrame(np.zeros((len(os.listdir(img_dir)), 13)))
        df.columns = ["imgpath", "protest", "violence", "sign", "photo",
                      "fire", "police", "children", "group_20", "group_100",
                      "flag", "night", "shouting"]
        df['imgpath'] = imgpaths
        df.iloc[:,1:] = np.concatenate(outputs)
        df.sort_values(by = 'imgpath', inplace=True)
        return df

def main():

    # load trained model
    print("*** loading model from {model}".format(model = args.model))
    # model = modified_resnet34()
    # model = modified_resnet50()
    if args.cuda:
        pass
        # model = model.cuda()
    with open(args.model) as f:
        
        model1 = torch.load('args.model.pth', 'cuda')
        print(type(model1))
        # model1 = model1["state_dict"]
        # print(model1.keys())
        # print(len(model1.keys()))
        # new_keys = ["layer1.2.conv1.weight", "layer1.2.bn1.weight", "layer1.2.bn1.bias", "layer1.2.bn1.running_mean", "layer1.2.bn1.running_var", "layer1.2.conv2.weight", "layer1.2.bn2.weight", "layer1.2.bn2.bias", "layer1.2.bn2.running_mean", "layer1.2.bn2.running_var", "layer2.2.conv1.weight", "layer2.2.bn1.weight", "layer2.2.bn1.bias", "layer2.2.bn1.running_mean", "layer2.2.bn1.running_var", "layer2.2.conv2.weight", "layer2.2.bn2.weight", "layer2.2.bn2.bias", "layer2.2.bn2.running_mean", "layer2.2.bn2.running_var", "layer2.3.conv1.weight", "layer2.3.bn1.weight", "layer2.3.bn1.bias", "layer2.3.bn1.running_mean", "layer2.3.bn1.running_var", "layer2.3.conv2.weight", "layer2.3.bn2.weight", "layer2.3.bn2.bias", "layer2.3.bn2.running_mean", "layer2.3.bn2.running_var", "layer3.2.conv1.weight", "layer3.2.bn1.weight", "layer3.2.bn1.bias", "layer3.2.bn1.running_mean", "layer3.2.bn1.running_var", "layer3.2.conv2.weight", "layer3.2.bn2.weight", "layer3.2.bn2.bias", "layer3.2.bn2.running_mean", "layer3.2.bn2.running_var", "layer3.3.conv1.weight", "layer3.3.bn1.weight", "layer3.3.bn1.bias", "layer3.3.bn1.running_mean", "layer3.3.bn1.running_var", "layer3.3.conv2.weight", "layer3.3.bn2.weight", "layer3.3.bn2.bias", "layer3.3.bn2.running_mean", "layer3.3.bn2.running_var", "layer3.4.conv1.weight", "layer3.4.bn1.weight", "layer3.4.bn1.bias", "layer3.4.bn1.running_mean", "layer3.4.bn1.running_var", "layer3.4.conv2.weight", "layer3.4.bn2.weight", "layer3.4.bn2.bias", "layer3.4.bn2.running_mean", "layer3.4.bn2.running_var", "layer3.5.conv1.weight", "layer3.5.bn1.weight", "layer3.5.bn1.bias", "layer3.5.bn1.running_mean", "layer3.5.bn1.running_var", "layer3.5.conv2.weight", "layer3.5.bn2.weight", "layer3.5.bn2.bias", "layer3.5.bn2.running_mean", "layer3.5.bn2.running_var", "layer4.2.conv1.weight", "layer4.2.bn1.weight", "layer4.2.bn1.bias", "layer4.2.bn1.running_mean", "layer4.2.bn1.running_var", "layer4.2.conv2.weight", "layer4.2.bn2.weight", "layer4.2.bn2.bias", "layer4.2.bn2.running_mean", "layer4.2.bn2.running_var"]
        # print(model["state_dict"].keys)
        # print(len(new_keys))
        # for key,n_key in zip(model1.keys(),new_keys):
        #     model1[n_key] = model1.pop(key)
        model = model1
        # model.load_state_dict(torch.load(args.model)['state_dict'])
        # checkpoint = torch.load('checkpoint.pth.tar')
        # model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
    print("*** calculating the model output of the images in {img_dir}"
            .format(img_dir = args.img_dir))

    # calculate output
    df = eval_one_dir(args.img_dir, model)

    # write csv file
    df.to_csv(args.output_csvpath, index = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir",
                        type=str,
                        default="UCLA-protest/img/test",
                        required = False,
                        help = "image directory to calculate output"
                        "(the directory must contain only image files)"
                        )
    parser.add_argument("--output_csvpath",
                        type=str,
                        default = "result.csv",
                        help = "path to output csv file"
                        )
    parser.add_argument("--model",
                        type=str,
                        default="checkpoint.pth.tar",
                        required = False,
                        help = "model path"
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
                        default = 32,
                        help = "batch size",
                        )
    args = parser.parse_args()
    args.cuda = True

    main()

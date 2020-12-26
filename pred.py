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
        model1 = torch.load(args.model, 'cuda')
        print(type(model1))
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
                        default = "single_perceptron_result.csv",
                        help = "path to output csv file"
                        )
    parser.add_argument("--model",
                        type=str,
                        default="output/single_perceptron_40ep/single_perceptron_best.pth", #checkpoint.pth.tar
                        required = False,
                        help = "model path eg:output/single_perceptron_40ep/single_perceptron_best.pth"
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

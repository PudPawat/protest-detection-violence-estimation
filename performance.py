import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns # I love this package!
sns.set_style('white')

import torch

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import scipy.stats as stats
def plot_roc(attr, target, pred):
    """Plot a ROC curve and show the accuracy score and the AUC"""
    fig, ax = plt.subplots()
    auc = roc_auc_score(target, pred)
    acc = accuracy_score(target, (pred >= 0.5).astype(int))
    fpr, tpr, _ = roc_curve(target, pred)
    plt.plot(fpr, tpr, lw = 2, label = attr.title())
    plt.legend(loc = 4, fontsize = 15)
    plt.title(('ROC Curve for {attr} (Accuracy = {acc:.3f}, AUC = {auc:.3f})'
               .format(attr = attr.title(), acc= acc, auc = auc)),
              fontsize = 15)
    plt.xlabel('False Positive Rate', fontsize = 15)
    plt.ylabel('True Positive Rate', fontsize = 15)
    plt.show()
    return fig

def main():
    # load check point
    model_path = args.model_dir
    checkpoint = torch.load(model_path)
    # print("ch",checkpoint)
    loss_history_train = checkpoint['loss_history_train']
    loss_history_val = checkpoint['loss_history_val']
    loss_his_val = [] # change type tensor to numpy for plotting
    for loss in loss_history_val:
        loxx = []
        for los in loss:
            loxx.append(los.cpu().numpy())
        loss_his_val.append(loxx)

    print(type(loss_history_val[0][0]))
    print(loss_history_train[0][0])
    loss_train = [np.mean(l) for l in loss_history_train]
    loss_val = [np.mean(l) for l in loss_his_val]
    plt.plot(loss_train, label = 'Train Loss')
    plt.plot(loss_val, label = 'Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Trend')
    plt.legend()
    plt.show()

    # load prediction
    df_pred = pd.read_csv(args.csv_dir)
    df_pred['imgpath'] = df_pred['imgpath'].apply(os.path.basename)

    # load target
    test_label_path = 'UCLA-protest/annot_test.txt'
    df_target = pd.read_csv(test_label_path, delimiter= '\t')
    # plot ROC curve for protest
    attr = "protest"
    target = df_target[attr]
    pred = df_pred[attr]
    fig = plot_roc(attr, target, pred)
    fig.savefig(os.path.join('output', attr+'.png'))
    # plot ROC curves for visual attributes
    for attr in df_pred.columns[3:]:
        target = df_target[attr]
        pred = df_pred[attr][target != '-']
        target = target[target != '-'].astype(int)
        fig = plot_roc(attr, target, pred)
        fig.savefig(os.path.join(args.save_dir, attr+'.png'))


    attr = 'violence'
    pred = df_pred[df_target['protest'] == 1][attr].tolist()
    target = df_target[df_target['protest'] == 1][attr].astype(float).tolist()
    fig, ax = plt.subplots()
    plt.scatter(target, pred, label = attr.title())
    plt.xlim([-.05,1.05])
    plt.ylim([-.05,1.05])
    plt.xlabel('Annotation', fontsize = 15)
    plt.ylabel('Predicton', fontsize = 15)
    corr, pval = stats.pearsonr(target, pred)
    plt.title(('Scatter Plot for {attr} (Correlation = {corr:.3f})'
                .format(attr = attr.title(), corr= corr)), fontsize = 15)
    plt.show()
    fig.savefig(os.path.join('files', attr+'.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir",
                        type=str,
                        default="output/resnet34_40ep/resnet34_best_result.csv",
                        required = False,
                        help = "image directory to calculate output"
                        "(the directory must contain only image files)"
                        )
    parser.add_argument("--model_dir",
                        type=str,
                        default="output/resnet34_40ep/checkpoint.pth.tar",
                        required = False,
                        help = "image directory to calculate output"
                        "(the directory must contain only image files)"
                        )
    parser.add_argument("--save_dir",
                        type=str,
                        default="output/single_perceptron_40ep/",
                        required = False,
                        help = "image directory to calculate output"
                        "(the directory must contain only image files)"
                        )


    args = parser.parse_args()


    main()
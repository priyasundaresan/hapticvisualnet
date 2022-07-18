from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import cv2
import os
import cmath
import math
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from config import *
from src.model import HapticVisualNet
from src.dataset import HapticVisualDataset, transform
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def run_inference(model, img, img_t, force_t, label_t, idx, output_dir='vis'):
    pred_label_logits = model(img_t.cuda().unsqueeze(0).float(), force_t.cuda().unsqueeze(0).float())
    pred_label_logits = pred_label_logits.detach().cpu().numpy()
    
    force_np = force_t.view(-1, 1).detach().cpu().numpy()
    label_np = label_t.item()

    pred_label = np.argmax(softmax(pred_label_logits))
    return pred_label, label_np

def get_confusion(test_dataset, model):
    torch.cuda.set_device(0)
    model = model.cuda()
    model.eval()
    y_pred = []
    y_true = []

    for idx, (img_np, img_t, label_t, force_t) in enumerate(test_dataset):
        pred_label, label_np = run_inference(model, img_np, img_t, force_t, label_t, idx, output_dir)
        y_pred.extend(np.array([pred_label]))
        y_true.extend(np.array([label_np]))

    classes = ('VS', 'AS')

    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix)*2, index=[i for i in classes],
                                 columns = [i for i in classes])
    #plt.figure(figsize=(4,4))
    #sn.heatmap(df_cm, annot=True)
    #plt.savefig('output.png')

    return df_cm

def plot_multiple_confusion_matrices(df, df2, df3):
    vmin = min(df.values.min(), df2.values.min(), df3.values.min())
    vmax = max(df.values.max(), df2.values.max(), df3.values.max())

    fig, axs = plt.subplots(nrows=4, gridspec_kw=dict(height_ratios=[4,4,4,0.2]))

    sn.heatmap(df, annot=True, cbar=False, ax=axs[0], vmin=vmin, vmax=vmax)
    sn.heatmap(df2, annot=True, cbar=False, ax=axs[1], vmin=vmin, vmax=vmax)
    sn.heatmap(df3, annot=True, cbar=False, ax=axs[2], vmin=vmin, vmax=vmax)
    #sns.heatmap(df2, annot=True, yticklabels=False, cbar=False, ax=axs[1], vmin=vmin, vmax=vmax)

    fig.colorbar(axs[1].collections[0], cax=axs[3])

    plt.savefig('output.png')
    #plt.show()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    dataset_dir = 'hapticnet_dset_6_10/test'

    use_haptic = True
    use_rgb = True

    if use_haptic and use_rgb:
        identifier = 'hapticvisual'
    elif use_haptic:
        identifier = 'haptic'
    elif use_rgb:
        identifier = 'visual'


    hapticvis_model = HapticVisualNet(use_haptic=True, use_rgb=True, out_classes=num_classes)
    hapticvis_model.load_state_dict(torch.load('/host/checkpoints/corl_nets/hapticvis.pth')) 

    haptic_model = HapticVisualNet(use_haptic=True, use_rgb=False, out_classes=num_classes)
    haptic_model.load_state_dict(torch.load('/host/checkpoints/corl_nets/haptic.pth')) 

    vis_model = HapticVisualNet(use_haptic=False, use_rgb=True, out_classes=num_classes)
    vis_model.load_state_dict(torch.load('/host/checkpoints/corl_nets/vis.pth')) 

    #model = hapticvis_model
    #model = vis_model
    model = haptic_model

    output_dir = 'vis'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    workers=0
    test_dataset = HapticVisualDataset('/host/datasets/%s'%dataset_dir, transform, num_classes)

    #sns.set(font="Palatino", style="white", font_scale=1)

    df1 = get_confusion(test_dataset, hapticvis_model)
    df2 = get_confusion(test_dataset, vis_model)
    df3 = get_confusion(test_dataset, haptic_model)

    df1.to_csv('hapticvis.csv')
    df2.to_csv('vis.csv')
    df3.to_csv('haptic.csv')

    #plot_multiple_confusion_matrices(df1, df2, df3)


    #for idx, (img_np, img_t, label_t, force_t) in enumerate(test_dataset):
    #    pred_label, label_np = run_inference(model, img_np, img_t, force_t, label_t, idx, output_dir)
    #    if pred_label==0:
    #        vertical_skewer_correct += int(pred_label==label_np)
    #        vertical_skewer_total += 1
    #    if pred_label==1:
    #        angled_skewer_correct += int(pred_label==label_np)
    #        angled_skewer_total += 1
    #    if pred_label==2:
    #        fail_correct += int(pred_label==label_np)
    #        fail_total += 1
    #    print("Annotating %06d"%idx)
    #print('Vert Skew Accuracy %:', vertical_skewer_correct/vertical_skewer_total)
    #print('Ang Skew Accuracy %:', angled_skewer_correct/angled_skewer_total)
    #print('Overall Accuracy %:', (fail_correct+vertical_skewer_correct+angled_skewer_correct)/(vertical_skewer_total+angled_skewer_total+fail_total))

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

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def run_inference(model, img, img_t, force_t, label_t, idx, output_dir='vis'):
    pred_label_logits = model(img_t.cuda().unsqueeze(0).float(), force_t.cuda().unsqueeze(0).float())
    pred_label_logits = pred_label_logits.detach().cpu().numpy()
    
    force_np = force_t.view(-1, 1).detach().cpu().numpy()
    label_np = label_t.item()

    mapping = {0: 'Vert. Skew', 1: 'Ang. Skew', 2: 'Fail'}
    pred_label = np.argmax(softmax(pred_label_logits))
    fig = plt.figure()
    fig.add_subplot(121)
    if pred_label == label_np:
        color = 'green'
    elif pred_label == 0 and label_np == 1 or pred_label == 1 and label_np == 0:
        color = 'red'
    else:
        color = 'orange'
    plt.title('Pred Label: %s\nActual Label: %s'%(mapping[pred_label], mapping[label_np]), color=color)
    plt.imshow(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB))
    fig.add_subplot(122)
    plt.title('Force')
    plt.plot(force_np)
    plt.ylim(0,12)
    plt.savefig('%s/%03d.jpg'%(output_dir, idx))
    return pred_label, label_np
    

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    use_haptic = True
    use_rgb = False
    if use_haptic and use_rgb:
        identifier = 'hapticvisual'
    elif use_haptic:
        identifier = 'haptic'
    elif use_rgb:
        identifier = 'visual'
    model = HapticVisualNet(use_haptic=use_haptic, use_rgb=use_rgb)
    #model.load_state_dict(torch.load('/host/checkpoints/hapticnet_dset_v0/model_2_1_19.pth'))
    model.load_state_dict(torch.load('/host/checkpoints/hapticnet_dset_v1_%s/model_2_1_18.pth'%identifier))
    torch.cuda.set_device(0)
    model = model.cuda()
    model.eval()
    output_dir = 'vis'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    workers=0
    dataset_dir = 'hapticnet_dset_v1'
    test_dataset = HapticVisualDataset('/host/datasets/%s/test'%dataset_dir, transform)

    vertical_skewer_correct = 0
    vertical_skewer_total = 0
    angled_skewer_correct = 0
    angled_skewer_total = 0
    fail_correct = 0
    fail_total = 0

    for idx, (img_np, img_t, label_t, force_t) in enumerate(test_dataset):
        pred_label, label_np = run_inference(model, img_np, img_t, force_t, label_t, idx, output_dir)
        if pred_label==0:
            vertical_skewer_correct += int(pred_label==label_np)
            vertical_skewer_total += 1
        if pred_label==1:
            angled_skewer_correct += int(pred_label==label_np)
            angled_skewer_total += 1
        if pred_label==2:
            fail_correct += int(pred_label==label_np)
            fail_total += 1
        print("Annotating %06d"%idx)
    print('Vert Skew Accuracy %:', vertical_skewer_correct/vertical_skewer_total)
    print('Ang Skew Accuracy %:', angled_skewer_correct/angled_skewer_total)
    print('Fail Accuracy %:', fail_correct/fail_total)
    print('Overall Accuracy %:', (fail_correct+vertical_skewer_correct+angled_skewer_correct)/(vertical_skewer_total+angled_skewer_total+fail_total))

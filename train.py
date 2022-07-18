import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from config import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
from src.model import HapticVisualNet
from src.dataset import HapticVisualDataset, transform

softmax = nn.Softmax(dim=1)
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def forward(sample_batched, model):
    _, img, label, force = sample_batched
    img = Variable(img.cuda() if use_cuda else img)
    force = Variable(force.cuda() if use_cuda else force)
    pred_labels_logits = model.forward(img.float(), force.float())
    pred_labels_logits = pred_labels_logits.view(-1,num_classes)
    pred_labels_probs = softmax(pred_labels_logits)
    pred_label = torch.argmax(pred_labels_probs, dim=1)
    label = label.view(-1)
    correct = (pred_label==label).sum()
    cls_loss = clsLoss(pred_labels_logits, label)
    return cls_loss, correct

def fit(train_data, test_data, model, epochs, checkpoint_path = ''):
    train_accs = []
    train_losses = []
    test_accs = []
    test_losses = []
    for epoch in range(epochs):
        train_cls_loss = 0.0
        train_cls_correct = 0.0
        for i_batch, sample_batched in enumerate(train_data):
            optimizer_cls.zero_grad()
            cls_loss, correct = forward(sample_batched, model)
            cls_loss.backward(retain_graph=True)
            optimizer_cls.step()
            train_cls_loss += cls_loss.item()
            train_cls_correct += correct.item()
            print('[%d, %5d] cls loss: %.3f, cls acc: %.3f' % \
	           (epoch + 1, \
	            i_batch + 1, \
                    cls_loss.item(), \
                    correct.item()/batch_size), \
                    end='')
            print('\r', end='')
        train_loss = train_cls_loss/i_batch
        train_acc = train_cls_correct/((i_batch+1)*batch_size)
        print('%d: train cls loss:'%(epoch), train_loss)
        print('%d: train cls accuracy:'%(epoch), train_acc)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        test_cls_loss = 0.0
        test_cls_correct = 0.0
        for i_batch, sample_batched in enumerate(test_data):
            cls_loss, correct = forward(sample_batched, model)
            test_cls_loss += cls_loss.item()
            test_cls_correct += correct.item()
        test_loss = test_cls_loss/i_batch
        test_acc = test_cls_correct/((i_batch+1)*batch_size)
        print('%d: test cls loss:'%(epoch), test_loss)
        print('%d: test cls accuracy:'%(epoch), test_acc)
        torch.save(model.state_dict(), checkpoint_path + '/model_' + str(epoch) + '_%.2f_%.2f.pth'%(test_loss, test_acc))
        test_losses.append(test_loss)
        test_accs.append(test_acc)
    return train_losses, train_accs, test_losses, test_accs

# dataset
workers=0
#dataset_dir = 'hapticnet_dset_v2'
#dataset_dir = 'hapticvis_dset_v0v2_combo'
#dataset_dir = 'hapticnet_dset_cheese'
#dataset_dir = 'hapticvis_dset_combo'
#dataset_dir = 'hapticnet_dset_6_5_22'
dataset_dir = 'hapticnet_dset_6_10'
output_dir = 'checkpoints'

parser = argparse.ArgumentParser()
parser.add_argument('--use_haptic', action='store_true')
parser.add_argument('--use_rgb', action='store_true')
args = parser.parse_args()

#use_haptic = True
#use_rgb = False

use_haptic = args.use_haptic
use_rgb = args.use_rgb

if use_haptic and use_rgb:
    identifier = 'hapticvisual_nodo'
elif use_haptic:
    identifier = 'haptic_nodo'
elif use_rgb:
    identifier = 'visual_nodo'

save_dir = os.path.join(output_dir, dataset_dir + '_' + identifier)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

train_dataset = HapticVisualDataset('/host/datasets/%s/train'%dataset_dir, transform, num_classes)
train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
test_dataset = HapticVisualDataset('/host/datasets/%s/test'%dataset_dir, transform, num_classes)
test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
use_cuda = torch.cuda.is_available()

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
if use_cuda:
    torch.cuda.set_device(0)

# model
#model = HapticVisualNet().cuda()
model = HapticVisualNet(use_haptic=use_haptic, use_rgb=use_rgb, out_classes=num_classes).cuda()

# optimizer
optimizer_cls = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1.0e-4)

print('dataset weights', train_dataset.weights)
clsLoss = nn.CrossEntropyLoss(weight=train_dataset.weights.cuda())

train_losses, train_accs, test_losses, test_accs = fit(train_data, test_data, model, epochs=epochs, checkpoint_path=save_dir)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
ax1.plot(train_losses)
ax1.plot(test_losses)
ax1.legend(['train', 'test'])
ax1.set_title('Losses')
ax2.plot(train_accs)
ax2.plot(test_accs)
ax1.legend(['train', 'test'])
ax2.set_title('Accs')
plt.savefig('%s/stats.jpg'%save_dir)

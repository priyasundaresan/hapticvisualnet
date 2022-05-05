import pickle
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
    pred_labels_logits = pred_labels_logits.view(-1,3)
    pred_labels_probs = softmax(pred_labels_logits)
    pred_label = torch.argmax(pred_labels_probs, dim=1)
    label = label.view(-1)
    correct = (pred_label==label).sum()
    cls_loss = clsLoss(pred_labels_logits, label)
    return cls_loss, correct

def fit(train_data, test_data, model, epochs, checkpoint_path = ''):
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
        print('%d: train cls loss:'%(epoch), train_cls_loss/i_batch)
        print('%d: train cls accuracy:'%(epoch), train_cls_correct/((i_batch+1)*batch_size))
        
        test_cls_loss = 0.0
        test_cls_correct = 0.0
        for i_batch, sample_batched in enumerate(test_data):
            cls_loss, correct = forward(sample_batched, model)
            test_cls_loss += cls_loss.item()
            test_cls_correct += correct.item()
        print('%d: test cls loss:'%(epoch), test_cls_loss/i_batch)
        print('%d: test cls accuracy:'%(epoch), test_cls_correct/((i_batch+1)*batch_size))
        torch.save(model.state_dict(), checkpoint_path + '/model_2_1_' + str(epoch) + '.pth')

# dataset
workers=0
dataset_dir = 'hapticnet_dset_v1'
output_dir = 'checkpoints'
use_haptic = False
use_rgb = True

if use_haptic and use_rgb:
    identifier = 'hapticvisual'
elif use_haptic:
    identifier = 'haptic'
elif use_rgb:
    identifier = 'visual'

save_dir = os.path.join(output_dir, dataset_dir + '_' + identifier)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

train_dataset = HapticVisualDataset('/host/datasets/%s/train'%dataset_dir, transform)
train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
test_dataset = HapticVisualDataset('/host/datasets/%s/test'%dataset_dir, transform)
test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
use_cuda = torch.cuda.is_available()

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
if use_cuda:
    torch.cuda.set_device(0)

# model
#model = HapticVisualNet().cuda()
model = HapticVisualNet(use_haptic=use_haptic, use_rgb=use_rgb).cuda()

# optimizer
optimizer_cls = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1.0e-4)

print('dataset weights', train_dataset.weights)
clsLoss = nn.CrossEntropyLoss(weight=train_dataset.weights.cuda())

fit(train_data, test_data, model, epochs=epochs, checkpoint_path=save_dir)

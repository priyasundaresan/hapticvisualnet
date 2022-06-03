import torch
import cv2
import time
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, utils
import numpy as np
import pickle
import os
from datetime import datetime

transform = transforms.Compose([transforms.ToTensor()])

class HapticVisualDataset(Dataset):
	def __init__(self, dataset_dir, transform, num_classes, num_haptic_readings=26, width=200, height=200):
		self.img_width = 200
		self.img_height = 200
		self.transform = transform
		self.imgs = []
		self.forces = []
		self.labels = []
		self.weights = torch.Tensor([0 for _ in range(num_classes)])
		force_folder = os.path.join(dataset_dir, 'force')
		labels_folder = os.path.join(dataset_dir, 'labels')
		img_folder = os.path.join(dataset_dir, 'images')
		for i in range(len(os.listdir(labels_folder))-1):
                    label = np.array([np.load(os.path.join(labels_folder, '%05d.npy'%i))])-1
                    self.weights[label] += 1
                    force = np.array([np.load(os.path.join(force_folder, '%05d.npy'%i))]).squeeze()
                    if len(force) > num_haptic_readings:
                        force = force[len(force)-num_haptic_readings:]
                    else:
                        pad = np.array([force[0] for _ in range(num_haptic_readings - len(force))])
                        force = np.hstack((pad, force))
                    force = force.reshape(1,-1)
                    self.imgs.append(os.path.join(img_folder, '%05d.jpg'%i))
                    self.forces.append(torch.from_numpy(force).cuda())
                    self.labels.append(torch.from_numpy(label).cuda())
		self.weights /= self.weights.sum()

	def __getitem__(self, index):  
		img_np = cv2.imread(self.imgs[index])
		img = self.transform(img_np).cuda().double()
		label = self.labels[index]
		force = self.forces[index]
		return img_np, img, label, force
    
	def __len__(self):
		return len(self.labels)

if __name__ == '__main__':
	dset = HapticVisualDataset('/host/datasets/hapticnet_dset_v0/train', transform)
	img_np, img, label, force = dset[0]
	print(img.shape, label.shape, force.shape, label, force)

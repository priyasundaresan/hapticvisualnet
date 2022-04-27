import os
import xml.etree.cElementTree as ET
from xml.dom import minidom
import random
import colorsys
import numpy as np
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
import matplotlib.pyplot as plt

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

IMG_AUGS = [ 
    iaa.LinearContrast((0.75, 1.25), per_channel=0.25), 
    iaa.Add((-20, 20), per_channel=False),
    iaa.AddToBrightness((-10, 30)),
    iaa.GammaContrast((0.85, 1.15)),
    iaa.GaussianBlur(sigma=(0.0, 0.6)),
    iaa.MultiplySaturation((0.85, 1.15)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.0125*255)),
    iaa.flip.Fliplr(0.5),
    sometimes(iaa.Affine(
                scale={"x": (1.0, 1.1), "y": (1.0, 1.1)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)}, # translate by -20 to +20 percent (per axis)
                rotate=(-5, 5), 
                shear=(-5, 5), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(100,150), # if mode is constant, use a cval between 0 and 255
                mode=['constant']
                #mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            ))
    ]

seq = iaa.Sequential(IMG_AUGS, random_order=True) 

def augment(img, force, label, img_dir, force_dir, label_dir, new_idx, show=False):
    img_aug = seq(image=img)
    if show:
        cv2.imshow('img', img_aug)
        cv2.waitKey(0)
        fig = plt.figure()
        fig.add_subplot(121)
        plt.title('image')
        plt.imshow(cv2.cvtColor(img_aug, cv2.COLOR_BGR2RGB))
        fig.add_subplot(122)
        plt.title('force ')
        plt.plot(force)
        plt.show()
    print(label, force)
    np.save(os.path.join(label_dir, '%05d.npy'%new_idx), label)
    np.save(os.path.join(force_dir, '%05d.npy'%new_idx), force)
    cv2.imwrite(os.path.join(img_dir, "%05d.jpg"%new_idx), img_aug)

if __name__ == '__main__':
    label_dir = 'labels'
    img_dir = 'images'
    force_dir = 'force'

    orig_len = len(os.listdir(img_dir))
    idx = orig_len
    num_augs_per_img = 8

    for i in range(orig_len):
        img = cv2.imread(os.path.join(img_dir, '%05d.jpg'%i))
        force = np.load(os.path.join(force_dir, '%05d.npy'%i))
        label = np.load(os.path.join(label_dir, '%05d.npy'%i))

        for _ in range(num_augs_per_img):
            #augment(img, force, label, img_dir, force_dir, label_dir, idx+i, show=True)
            augment(img, force, label, img_dir, force_dir, label_dir, idx+i, show=False)
            idx += 1
        idx -= 1

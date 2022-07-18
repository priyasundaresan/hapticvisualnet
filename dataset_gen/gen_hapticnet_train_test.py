import numpy  as np
import cv2
from random import shuffle
import os
import matplotlib.pyplot as plt

def preprocess(img):
    img_new = img.copy()
    u,v = (650,350)
    H,W = 350,350
    img_new = img_new[v:v+H, u:u+W]
    img_new = cv2.resize(img_new, (200,200))
    return img_new

if __name__ == '__main__':
    dataset_dir = os.getcwd()
    output_train_dir = os.path.join(dataset_dir, 'train')
    output_train_img_dir = os.path.join(output_train_dir, 'images')
    output_train_force_dir = os.path.join(output_train_dir, 'force')
    output_val_dir = os.path.join(dataset_dir, 'test')
    output_val_img_dir = os.path.join(output_val_dir, 'images')
    output_val_force_dir = os.path.join(output_val_dir, 'force')
    dirs = [output_train_dir, output_train_img_dir, output_train_force_dir, output_val_dir, output_val_img_dir, output_val_force_dir]
    for d in dirs:
        if not os.path.exists(d):
            os.mkdir(d)

    #inp_data_dir = 'hapticnet_data_v0'
    #inp_data_dir = 'hapticnet_data_v2'
    inp_data_dir = 'hapticnet_data_combo'

    inp_img_dir = os.path.join(inp_data_dir, 'images')
    inp_force_dir = os.path.join(inp_data_dir, 'force')
    file_idxs = list(range(len(os.listdir(inp_img_dir))))
    print(file_idxs)

    n_total_images = len(file_idxs)

    train_idx = 0
    val_idx = 0
    split_idx = int(n_total_images*0.8)

    shuffle(file_idxs)

    min_len_force = float('inf')
    for i, idx in enumerate(file_idxs):
        img = cv2.imread("%s/%03d.jpg"%(inp_img_dir, idx))
        img = preprocess(img)
        force = np.load("%s/%03d.npy"%(inp_force_dir, idx))[-26:]
        if np.isnan(np.sum(force)):
            continue
        if len(force) < min_len_force:
            min_len_force = len(force)

        #cv2.imshow('img', img)
        #cv2.waitKey(0)

        #fig = plt.figure()
        #fig.add_subplot(121)
        #plt.title('image')
        #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #
        #fig.add_subplot(122)
        #plt.title('force ')
        #plt.plot(force)
        #plt.show()
        
        if i < split_idx:
            cv2.imwrite(os.path.join(output_train_img_dir, '%05d.jpg'%train_idx), img)
            np.save(os.path.join(output_train_force_dir, '%05d.npy'%train_idx), force)
            train_idx += 1
        else:
            cv2.imwrite(os.path.join(output_val_img_dir, '%05d.jpg'%val_idx), img)
            np.save(os.path.join(output_val_force_dir, '%05d.npy'%val_idx), force)
            val_idx += 1
        if i == n_total_images:
            break

print(min_len_force)

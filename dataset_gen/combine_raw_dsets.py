import numpy as np
import cv2
import os

def combine_dsets(output_dir, inp_dirs):
    output_img_dir = os.path.join(output_dir, 'images')
    output_force_dir = os.path.join(output_dir, 'force')

    ctr = 0
    for dset_dir in inp_dirs:
        img_dir = os.path.join(dset_dir, 'images')
        force_dir = os.path.join(dset_dir, 'force')
        for idx, fn in enumerate(sorted(os.listdir(img_dir))):
            img = cv2.imread(os.path.join(img_dir, fn))
            force = np.load(os.path.join(force_dir, '%03d.npy'%idx))
            cv2.imwrite(os.path.join(output_img_dir, '%03d.jpg'%ctr), img)
            np.save(os.path.join(output_force_dir, '%03d.npy'%ctr), force)
            ctr += 1
            print(ctr)

if __name__ == '__main__':
    output_dir = 'hapticvis_dset_combo'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        os.mkdir(os.path.join(output_dir, 'images'))
        os.mkdir(os.path.join(output_dir, 'force'))

    inp_dirs = ['hapticnet_data_v0', 'hapticnet_data_v2']
    combine_dsets(output_dir, inp_dirs)
    

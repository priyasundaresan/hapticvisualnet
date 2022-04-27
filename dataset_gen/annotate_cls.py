import cv2
import numpy as np
import os

class ClsAnnotator:
    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.labels_map = {1: 'vertical skewer', 2:'angled skewer', 3:'failure'}

    def load_image(self, img):
        self.img = img
        self.vis = img.copy()

    def run(self, img):
        self.load_image(img)
        cv2.imshow('vis', self.vis)
        res = cv2.waitKey(0)
        self.label = int(chr(res%256))
        return self.label

if __name__ == '__main__':
    cls_selector = ClsAnnotator(num_classes=3)

    image_dir = 'images' # Should have images like 00000.jpg, 00001.jpg, ...
    force_dir = 'force' # Should have images like 00000.jpg, 00001.jpg, ...

    output_dir = 'real_data' # Will have real_data/images and real_data/keypoints

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    force_output_dir = os.path.join(output_dir, 'force')
    images_output_dir = os.path.join(output_dir, 'images')
    labels_output_dir = os.path.join(output_dir, 'labels')

    if not os.path.exists(force_output_dir):
        os.mkdir(force_output_dir)
    if not os.path.exists(images_output_dir):
        os.mkdir(images_output_dir)
    if not os.path.exists(labels_output_dir):
        os.mkdir(labels_output_dir)

    i = 0

    for f in sorted(os.listdir(image_dir)):
        print("Img %d"%i)
        image_path = os.path.join(image_dir, f)
        force_path = os.path.join(force_dir, '%05d.npy'%i)
        img = cv2.imread(image_path)
        force = np.load(force_path)

        image_outpath = os.path.join(images_output_dir, '%05d.jpg'%i)
        force_outpath = os.path.join(force_output_dir, '%05d.npy'%i)
        label_outpath = os.path.join(labels_output_dir, '%05d.npy'%i)

        img_label = cls_selector.run(img)

        print(force, cls_selector.labels_map[img_label])
        cv2.imwrite(image_outpath, img)
        np.save(force_outpath, force)
        np.save(label_outpath, img_label)
        i+= 1

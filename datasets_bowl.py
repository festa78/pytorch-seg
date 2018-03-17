import collections
import glob
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.io import imread
import torch
import torchvision
from torch.utils import data
from transform import HorizontalFlip, VerticalFlip

def default_loader(path):
    return Image.open(path)

class BowlDataSet(data.Dataset):
    def __init__(self, root, img_transform=None, label_transform=None):
        self.root = root
        self.mode = 'train'
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.h_flip = HorizontalFlip()
        self.v_flip = VerticalFlip()

        for mode in ['train', 'val']:
            image_basenames = [osp.basename(f) for f in glob.glob(osp.join(self.root, mode, 'rgb/*.png'))]
            for name in image_basenames:
                img_file = osp.join(self.root, mode, 'rgb', name)
                label_file = osp.join(self.root, mode, 'gt', name)
                self.files[mode].append({
                    "img": img_file,
                    "label": label_file
                })

    def __len__(self):
        return len(self.files[self.mode])

    def __getitem__(self, index):
        datafiles = self.files[self.mode][index]

        img_file = datafiles["img"]
        img = Image.open(img_file).convert('RGB')

        label_file = datafiles["label"]
        label = Image.open(label_file).convert("P")
        label_size = label.size

        if self.img_transform is not None:
            img_o = self.img_transform(img)
            imgs = [img_o]
        else:
            imgs = img

        if self.label_transform is not None:
            label_o = self.label_transform(label)
            labels = [label_o]
        else:
            labels = label

        return imgs, labels


class BowlTestSet(data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        # Get test IDs.
        self.test_ids = next(os.walk(self.root))[1]

        # Get test images and their sizes.
        self.img_names = []
        for n, id_ in enumerate(self.test_ids):
            self.img_names.append(osp.join(self.root, id_, 'images/', id_ + '.png'))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        filepath = self.img_names[index]
        img = imread(filepath)[...,:3]
        size = img.shape
        name = osp.basename(filepath)

        if self.transform is not None:
            img = self.transform(Image.fromarray(img))

        return img, name, size

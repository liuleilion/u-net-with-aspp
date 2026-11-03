import glob
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import os
from PIL import Image
import numpy as np
from torchvision import transforms
import nibabel as nib


class MyDataset(Dataset):
    def __init__(self, img_dir, gt_dir):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.img_list = os.listdir(img_dir)


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        gt_path = os.path.join(self.gt_dir, self.img_list[idx])

        image = Image.open(img_path).convert("L")
        gt = Image.open(gt_path).convert("L")

        gt = np.array(gt)
        gt_lv = (gt == 1).astype(np.float32)
        gt_rv = (gt == 2).astype(np.float32)
        gt_myo = (gt == 3).astype(np.float32)
        gt = np.stack([gt_lv, gt_rv, gt_myo], axis=0)


        image = self.transform(image)

        return image, gt
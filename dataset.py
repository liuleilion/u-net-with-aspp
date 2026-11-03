import glob
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import os
from PIL import Image
import numpy as np
from torchvision import transforms
import nibabel as nib
import sys


def encode_mask(gt):
    gt = np.array(gt)
    return np.stack([
        (gt == 255),   # LV  (ACDC label 3)
        (gt == 85),    # RV  (ACDC label 1)
        (gt == 170),   # Myo (ACDC label 2)
    ], axis=0).astype(np.float32)


class MyDataset(Dataset):
    def __init__(self, img_dir, gt_dir):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.img_list = os.listdir(img_dir)

        print(f"img_files size={len(self.img_list)}, mask_files size={len(self.img_list)}", file=sys.stderr)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        gt_path = os.path.join(self.gt_dir, self.img_list[idx])

        image = Image.open(img_path).convert("L")
        gt = Image.open(gt_path).convert("L")

        gt = encode_mask(gt)

        image = self.transform(image)

        return image, gt
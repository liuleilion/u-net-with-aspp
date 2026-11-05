import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import MyDataset, encode_mask
from model.UNet import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = './data'
IMG_SIZE = 512
WEIGHTS = 'unet-best.pth'


def dice_coefficient(pred, target, eps=1e-6):
    pred = pred.view(pred.size(0), pred.size(1), -1)
    target = target.view(target.size(0), target.size(1), -1)
    intersection = (pred * target).sum(dim=2)
    union = pred.sum(dim=2) + target.sum(dim=2)
    dice = (2. * intersection + eps) / (union + eps)
    return dice.mean(dim=1).mean().item()


def iou_coefficient(pred, target, eps=1e-6):
    pred = pred.view(pred.size(0), pred.size(1), -1)
    target = target.view(target.size(0), target.size(1), -1)
    intersection = (pred * target).sum(dim=2)
    union = pred.sum(dim=2) + target.sum(dim=2) - intersection
    return ((intersection + eps) / (union + eps)).mean().item()


def load_model(weights=WEIGHTS):
    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(weights, map_location=device))
    return model.to(device).eval()


def _prepare(image, gt):
    image = transforms.Resize((IMG_SIZE, IMG_SIZE))(image)
    gt = transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=Image.NEAREST)(gt)
    return image, gt


def test():
    img_dir = os.path.join(DATA_ROOT, 'test', 'Img')
    gt_dir = os.path.join(DATA_ROOT, 'test', 'GT')
    name = sorted(os.listdir(img_dir))[0]
    image_path = os.path.join(img_dir, name)
    gt_path = os.path.join(gt_dir, name)

    model = load_model()

    with torch.no_grad():
        image = Image.open(image_path).convert("RGB")
        gt = Image.open(gt_path).convert("L")
        image, gt = _prepare(image, gt)

        x = transforms.ToTensor()(image).unsqueeze(0).to(device)
        y = torch.from_numpy(encode_mask(gt)).unsqueeze(0).to(device)

        prob = torch.sigmoid(model(x))[0, 0]
        pred = (prob > 0.5).cpu().numpy().astype(bool)

        rgb = np.array(image)
        overlay = rgb.copy()
        overlay[pred] = (0.45 * rgb[pred] + 0.55 * np.array([255, 40, 40])).astype(np.uint8)

        Image.fromarray(overlay).save('result.jpg')
        Image.fromarray((pred * 255).astype(np.uint8)).save('result_mask.png')

        dice_score = dice_coefficient((prob > 0.5)[None, None].float(), y.float())
        print(f'Dice Coefficient {name}: {dice_score:.4f}')


def eval():
    model = load_model()
    test_dataset = MyDataset(os.path.join(DATA_ROOT, 'test', 'Img'),
                             os.path.join(DATA_ROOT, 'test', 'GT'),
                             img_size=IMG_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    dice_scores, iou_scores = [], []
    with torch.no_grad():
        for images, gts in test_loader:
            images = images.to(device)
            gts = gts.to(device)
            preds = (torch.sigmoid(model(images)) > 0.5).float()
            dice_scores.append(dice_coefficient(preds, gts))
            iou_scores.append(iou_coefficient(preds, gts))

    print(f'Average Dice: {np.mean(dice_scores):.4f}')
    print(f'Average IoU : {np.mean(iou_scores):.4f}')


if __name__ == '__main__':
    test()
    eval()

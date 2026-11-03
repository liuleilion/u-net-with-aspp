from dataset import *
from torchvision import transforms
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dice_coefficient(pred, target, eps=1e-6):
    pred = pred.view(pred.size(0), pred.size(1), -1)
    target = target.view(target.size(0), target.size(1), -1)
    intersection = (pred * target).sum(dim=2)
    union = pred.sum(dim=2) + target.sum(dim=2)
    dice = (2. * intersection + eps) / (union + eps)
    return dice.mean(dim=1).mean().item()


def test():
    image_path = './data/test/Img/patient001_01_1.png'
    gt_path  = './data/test/GT/patient001_01_1.png'
    model = torch.load('UNet.pth')
    model.eval()
    with torch.no_grad():
        image = Image.open(image_path).convert("L")
        gt = Image.open(gt_path).convert("L")

        transform = transforms.Compose([transforms.ToTensor()])
        image = transform(image).unsqueeze(0).to(device)

        gt = np.array(gt)
        gt_lv = (gt == 1).astype(np.float32)
        gt_rv = (gt == 2).astype(np.float32)
        gt_myo = (gt == 3).astype(np.float32)
        gt = np.stack([gt_lv, gt_rv, gt_myo], axis=0)
        gt = torch.tensor(gt).unsqueeze(0).to(device)

        # 模型推理
        output = model(image)
        prob = F.softmax(output, dim=1)
        pred_class = torch.argmax(prob, dim=1)

        # segment 结果保存
        pred = pred_class[0]  # H,W
        h, w = pred.shape
        segcolor = torch.zeros(h, w, 3, device=pred.device)

        # 类别1 → R通道
        mask_1 = (pred == 1).float()
        segcolor[:, :, 0] = mask_1

        # 类别2 → G通道
        mask_2 = (pred == 2).float()
        segcolor[:, :, 1] = mask_2

        image_seg = Image.fromarray(np.array(segcolor.detach().cpu() * 255, dtype=np.uint8))

        image_seg.save('result.jpg')


        # 计算Dice系数
        preds = torch.sigmoid(output) > 0.5
        dice_score = dice_coefficient(preds, gt)
        print(f'该图片的Dice Coefficient {image_path}: {dice_score}')



def eval(): #在整个测试集上评估模型
    model = torch.load('UNet.pth')
    model.eval()
    dice_scores = []
    test_dataset = MyDataset('./data/test/Img', './data/test/GT')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)
    with torch.no_grad():
        for images, gts in test_loader:
            images = images.to(device)
            gts = gts.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            dice_scores.append(dice_coefficient(preds, gts))
    avg_dice = np.mean(dice_scores)
    print(f'测试集上的Average Dice Coefficient为: {avg_dice}')


if __name__ == '__main__':
    test()
    eval()
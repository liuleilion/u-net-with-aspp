from dataset import *
from torchvision import transforms
from model.UNet import UNet
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
    
    model = UNet(in_channels=1, out_channels=3)
    model.load_state_dict(torch.load('unet-best.pth', map_location=device))
    model.to(device)

    model.eval()
    
    with torch.no_grad():
        image = Image.open(image_path).convert("L")
        gt = Image.open(gt_path).convert("L")

        transform = transforms.Compose([transforms.ToTensor()])
        image = transform(image).unsqueeze(0).to(device)

        gt = encode_mask(gt)
        gt = torch.tensor(gt).unsqueeze(0).to(device)

        # 模型推理（3 通道独立 sigmoid，背景=三通道全 0）
        output = model(image)
        pred = torch.sigmoid(output)[0] > 0.5  # [3,H,W]

        # segment 结果保存：LV→R, RV→G, Myo→B
        h, w = pred.shape[1], pred.shape[2]
        segcolor = torch.zeros(h, w, 3, device=pred.device)
        segcolor[:, :, 0] = pred[0].float()
        segcolor[:, :, 1] = pred[1].float()
        segcolor[:, :, 2] = pred[2].float()

        image_seg = Image.fromarray(np.array(segcolor.detach().cpu() * 255, dtype=np.uint8))

        image_seg.save('result.jpg')


        # 计算Dice系数
        preds = torch.sigmoid(output) > 0.5
        dice_score = dice_coefficient(preds, gt)
        print(f'Dice Coefficient {image_path}: {dice_score}')



def eval(): #在整个测试集上评估模型
    model = UNet(in_channels=1, out_channels=3)
    model.load_state_dict(torch.load('unet-best.pth', map_location=device))
    model.to(device)
    model.eval()
    dice_scores = []
    test_dataset = MyDataset('./data/test/Img', './data/test/GT')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
    with torch.no_grad():
        for images, gts in test_loader:
            images = images.to(device)
            gts = gts.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            dice_scores.append(dice_coefficient(preds, gts))
    avg_dice = np.mean(dice_scores)
    print(f'Average Dice Coefficient为: {avg_dice}')


if __name__ == '__main__':
    test()
    eval()
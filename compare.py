import argparse
import glob
import os
import re

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from dataset import encode_mask
from model.UNet import UNet


def natural_key(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]


def find_models(args):
    """返回待加载的权重列表。指定 --model-step 时，取 unet-best + 每 step 个 epoch 的权重。"""
    if args.model_step is None:
        return [args.weights]

    if args.model_step <= 0:
        raise ValueError('--model-step 必须为正整数')

    models = []
    pattern = os.path.join(args.model_dir, 'unet-epoch-*.pth')
    epochs = []
    for path in glob.glob(pattern):
        m = re.search(r'unet-epoch-(\d+)\.pth$', os.path.basename(path))
        if m and int(m.group(1)) % args.model_step == 0:
            epochs.append((int(m.group(1)), os.path.abspath(path)))
    epochs.sort()

    for _, path in epochs:
        models.append(path)

    if os.path.exists(args.weights):
        best = os.path.abspath(args.weights)
        if best not in models:
            models.append(best)
    else:
        print(f'[compare] warning: {args.weights} not found, skipped')

    if not models:
        raise FileNotFoundError(f'no matching weights in {args.model_dir}')
    return models


def colorize(mask):
    m = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
    m = (m > 0.5).astype(np.uint8)
    out = np.zeros((m.shape[1], m.shape[2], 3), dtype=np.uint8)
    out[..., 0] = m[0] * 255  # LV  -> R
    out[..., 1] = m[1] * 255  # RV  -> G
    out[..., 2] = m[2] * 255  # Myo -> B
    return Image.fromarray(out)


def dice_per_sample(pred, gt, eps=1e-6):
    pred = pred.float().view(pred.size(0), -1)
    gt = gt.float().view(gt.size(0), -1)
    inter = (pred * gt).sum(dim=1)
    union = pred.sum(dim=1) + gt.sum(dim=1)
    return (((2. * inter + eps) / (union + eps)).mean()).item()


def get_font(size=14):
    for path in [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf"]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def build_canvas(rows, model_names, title, out_path):
    """rows: [(name, image_rgb, gt_img, [(pred_img, dice), ...]), ...]"""
    widths, heights = [], []
    for _, image, gt_img, preds in rows:
        widths += [image.width, gt_img.width]
        heights += [image.height, gt_img.height]
        for pred_img, _ in preds:
            widths.append(pred_img.width)
            heights.append(pred_img.height)
    cw, ch = max(widths), max(heights)

    pad, title_h, header_h, name_h = 8, 28, 30, 20
    cols = 2 + len(model_names)
    width = pad + cols * (cw + pad)
    height = pad + title_h + header_h + len(rows) * (name_h + ch + pad)
    canvas = Image.new('RGB', (width, height), (25, 25, 28))
    draw = ImageDraw.Draw(canvas)

    font = get_font(16)
    small = get_font(13)

    tw = draw.textlength(title, font=font)
    draw.text(((width - tw) / 2, pad), title, font=font, fill=(255, 220, 120))

    header_y = pad + title_h
    headers = ['Image', 'GT (Mask)'] + model_names
    for c, header in enumerate(headers):
        hw = draw.textlength(header, font=small)
        x = pad + c * (cw + pad) + (cw - hw) / 2
        draw.text((x, header_y + 8), header, font=small, fill=(235, 235, 235))

    y = header_y + header_h
    for name, image, gt_img, preds in rows:
        draw.text((pad, y + 2), name, font=small, fill=(120, 200, 255))
        y += name_h

        canvas.paste(image, (pad, y))
        canvas.paste(gt_img, (pad + (cw + pad), y))

        for j, (pred_img, dice) in enumerate(preds):
            x = pad + (2 + j) * (cw + pad)
            canvas.paste(pred_img, (x, y))
            text = f'{dice:.4f}'
            tw2 = draw.textlength(text, font=small)
            draw.rectangle([x + 2, y + 2, x + 8 + tw2, y + 4 + 15], fill=(0, 0, 0))
            draw.text((x + 5, y + 3), text, font=small, fill=(255, 235, 120))

        y += ch + pad

    canvas.save(out_path, quality=95)


def main():
    ap = argparse.ArgumentParser(description="Visualize image / GT mask / predictions on one grid.")
    ap.add_argument('--img-dir', default='./data/test/Img')
    ap.add_argument('--gt-dir', default='./data/test/GT')
    ap.add_argument('--weights', default='unet-best.pth', help='默认加载的权重（--model-step 时作为必含项）')
    ap.add_argument('--model-dir', default='.', help='unet-epoch-*.pth 所在目录')
    ap.add_argument('--model-step', type=int, default=None,
                    help='模型 epoch 采样间隔，如 10 => unet-epoch-0/10/20... + unet-best')
    ap.add_argument('--step', type=int, default=10, help='图像采样间隔：每隔 step 张取一张')
    ap.add_argument('--start', type=int, default=0, help='从第几张开始采样')
    ap.add_argument('--max-samples', type=int, default=40, help='最多展示多少个样本')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--out', default='compare.jpg')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    files = sorted(os.listdir(args.img_dir), key=natural_key)
    selected = files[args.start::args.step][:args.max_samples]
    if not selected:
        print('[compare] no samples selected')
        return

    transform = transforms.Compose([transforms.ToTensor()])
    samples = []
    for name in selected:
        image = Image.open(os.path.join(args.img_dir, name)).convert('L')
        gt = Image.open(os.path.join(args.gt_dir, name)).convert('L')
        samples.append((name, image, torch.tensor(encode_mask(gt))))

    model_paths = find_models(args)
    model_names = [os.path.splitext(os.path.basename(p))[0] for p in model_paths]
    print(f'[compare] {len(samples)} samples, {len(model_paths)} model(s)')

    preds_by_model = []
    for model_path, model_name in zip(model_paths, model_names):
        model = UNet(in_channels=1, out_channels=3)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device).eval()

        preds = []
        dice_scores = []
        with torch.no_grad():
            for _, image, gt_mask in samples:
                x = transform(image).unsqueeze(0).to(device)
                pred = torch.sigmoid(model(x))[0] > args.threshold
                dice = dice_per_sample(pred, gt_mask.to(device))
                dice_scores.append(dice)
                preds.append((colorize(pred), dice))
        preds_by_model.append(preds)
        print(f'[compare] {model_name}: avg Dice {np.mean(dice_scores):.4f}')

    rows = []
    for i, (name, image, gt_mask) in enumerate(samples):
        rows.append((
            name,
            image.convert('RGB'),
            colorize(gt_mask),
            [preds_by_model[m][i] for m in range(len(model_paths))],
        ))

    build_canvas(rows, model_names, 'Image / GT / Prediction', args.out)
    print(f'[compare] saved -> {args.out}')


if __name__ == '__main__':
    main()

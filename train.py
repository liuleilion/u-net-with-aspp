import os

import configs.config_loader as cfg_loader
from model.UNet import *
from model.training import *
from dataset import *


def main():
    args = cfg_loader.get_config()
    model = UNet(in_channels=3, out_channels=1)

    train_dir = os.path.join(args.data_root, 'train')
    TrainDataset = MyDataset(os.path.join(train_dir, 'Img'),
                             os.path.join(train_dir, 'GT'),
                             img_size=args.img_size,
                             augment=True)

    trainer = Trainer(model, TrainDataset, args)
    trainer.train_model()


if __name__ == '__main__':
    main()

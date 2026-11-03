import configs.config_loader as cfg_loader
from model.UNet import *
from model.training import *
from dataset import *


def main():
    args = cfg_loader.get_config()
    model = UNet(in_channels=1, out_channels=3)
    TrainDataset = MyDataset('./data/train/Img', './data/train/GT')
    trainer = Trainer(model, TrainDataset, args)
    trainer.train_model()


if __name__ == '__main__':
    main()

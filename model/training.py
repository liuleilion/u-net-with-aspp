from __future__ import division
import torch
import numpy as np
from torch.utils import data
from model.myloss import  compute_weighted_loss


class Trainer(object):
    def __init__(self, model, TrainDataset, args):
        self.args = args
        self.model = model
        if torch.cuda.is_available():
            self.model.cuda()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Trainloader = data.DataLoader(dataset=TrainDataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(lr=1e-3, params=model.parameters())
        self.epochs = args.epochs
        self.start_epoch = args.start_epoch


    def train_model(self):
        self.model.train()
        for epoch in range(self.start_epoch,1 ):#self.epochs
            epoch_loss = 0
            for i, (images, gts) in enumerate(self.Trainloader):
                images = images.to(self.device)
                gts = gts.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, gts)

                # 处理gts：去掉通道维，再转long
                loss2 = compute_weighted_loss(outputs, gts)

                loss = loss+loss2
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

                if i % 3==0:
                    print(loss.item())
            print(f'Epoch {epoch + 1}/{(self.epochs-self.start_epoch)}, 损失函数的平均值为: {epoch_loss / len(self.Trainloader)}')

            if epoch%3 == 0:
                 torch.save(self.model, 'UNet.pth')

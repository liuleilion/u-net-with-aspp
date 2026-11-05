from __future__ import division
import torch
import numpy as np
from torch.utils import data
from model.myloss import dice_loss
import sys

class Trainer(object):
    def __init__(self, model, TrainDataset, args):
        self.args = args

        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.Trainloader = data.DataLoader(dataset=TrainDataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(lr=1e-3, params=model.parameters())

        self.epochs = args.epochs
        self.start_epoch = args.start_epoch


    def train_model(self):
        
        # 保留最优的epoch数据
        best_loss = float('inf')

        self.model.train()

        for epoch in range(self.start_epoch, self.epochs):
            
            epoch_loss = 0
            
            for i, (images, gts) in enumerate(self.Trainloader):
                
                images = images.to(self.device)
                gts = gts.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(images)

                loss1 = self.criterion(outputs, gts)
                loss2 = dice_loss(outputs, gts)
                loss = loss1 + loss2

                loss.backward()

                self.optimizer.step()

                epoch_loss += loss.item()

                print(f'epoch-{epoch}-{i}---loss:{loss:.4f}---loss1:{loss1:.4f}---loss2:{loss2:.4f}', file=sys.stderr)

            avg_loss = epoch_loss / len(self.Trainloader)
            print(f'Epoch {epoch}, average-loss: {avg_loss}', file=sys.stderr)

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), 'unet-best.pth')
                print(f'saved best model at epoch{epoch} with loss:{best_loss}', file=sys.stderr)

            if epoch%5 == 0:
                torch.save(self.model.state_dict(), f'unet-epoch-{epoch}.pth')

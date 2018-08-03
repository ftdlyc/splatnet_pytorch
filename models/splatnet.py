import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../utils"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils.permutohedral_lattice_layer import PermutohedralLattice


class SplatNetSegment(nn.Module):

    def __init__(self, class_nums, category_nums, pos_lambda=64, device_id=0, initial_weights=True):
        super(SplatNetSegment, self).__init__()

        self.class_nums = class_nums
        self.category_nums = category_nums
        self.device_id = device_id

        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 32, 1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(True)
        )
        self.pl1 = PermutohedralLattice(32, 64, 3, pos_lambda, bias=False)
        self.pl1_bn = nn.Sequential(
            nn.BatchNorm1d(64),
            nn.ReLU(True)
        )
        self.pl2 = PermutohedralLattice(64, 128, 3, pos_lambda / 2, bias=False)
        self.pl2_bn = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.ReLU(True)
        )
        self.pl3 = PermutohedralLattice(128, 256, 3, pos_lambda / 4, bias=False)
        self.pl3_bn = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.ReLU(True)
        )
        self.pl4 = PermutohedralLattice(256, 256, 3, pos_lambda / 8, bias=False)
        self.pl4_bn = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.ReLU(True)
        )
        self.pl5 = PermutohedralLattice(256, 256, 3, pos_lambda / 16, bias=False)
        self.pl5_bn = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.ReLU(True)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64 + 128 + 3 * 256 + category_nums, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Conv1d(256, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Conv1d(128, class_nums, 1)
        )

        if initial_weights:
            self.initialize_weights()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.01, weight_decay=0.001)
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 20, 0.5)

        self.cuda(device_id)

    def forward(self, x, labels, position):
        x = self.mlp1(x)
        x1 = self.pl1(x, position)
        x1 = self.pl1_bn(x1)
        x2 = self.pl2(x1, position)
        x2 = self.pl2_bn(x2)
        x3 = self.pl3(x2, position)
        x3 = self.pl3_bn(x3)
        x4 = self.pl4(x3, position)
        x4 = self.pl4_bn(x4)
        x5 = self.pl5(x4, position)
        x5 = self.pl5_bn(x5)
        index = labels.unsqueeze(1).repeat([1, x.size(2)]).unsqueeze(1)
        one_hot = torch.zeros([x.size(0), self.category_nums, x.size(2)])
        one_hot = one_hot.cuda(self.device_id)
        one_hot = one_hot.scatter_(1, index, 1)
        x = torch.cat([x1, x2, x3, x4, x5, one_hot], dim=1)
        x = self.mlp2(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, PermutohedralLattice):
                m.weights.data.normal_(0, 0.01)

    def loss(self, outputs, targets):
        return self.criterion(outputs, targets)

    def fit(self, dataloader, epoch):
        self.train()
        batch_loss = 0.
        epoch_loss = 0.
        batch_nums = 0
        if self.schedule is not None:
            self.schedule.step()

        print('----------epoch %d start train----------' % epoch)

        for batch_idx, (inputs, targets, labels) in enumerate(dataloader):
            inputs = inputs.cuda(self.device_id)
            targets = targets.cuda(self.device_id)
            labels = labels.cuda(self.device_id)
            position = inputs.transpose(1, 2).contiguous()
            self.optimizer.zero_grad()

            outputs = self(inputs, labels, position.detach())
            losses = self.loss(outputs, targets)
            losses.backward()
            self.optimizer.step()

            batch_loss += losses.item()
            epoch_loss += losses.item()
            batch_nums += 1
            if (batch_idx + 1) % 4 == 0:
                print('[%d, %5d] loss %.3f' % (epoch, batch_idx, batch_loss / 4))
                batch_loss = 0.

        print('-----------epoch %d end train-----------' % epoch)
        print('epoch %d loss %.3f' % (epoch, epoch_loss / batch_nums))

        return epoch_loss / batch_nums

    def score(self, dataloader):
        self.eval()
        correct = 0.
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets, labels) in enumerate(dataloader):
                inputs = inputs.cuda(self.device_id)
                targets = targets.cuda(self.device_id)
                labels = labels.cuda(self.device_id)
                position = inputs.transpose(1, 2).contiguous()

                outputs = self(inputs, labels, position.detach())
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0) * targets.size(1)
                correct += (predicted == targets).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

        return correct / total

#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision import transforms
import torch.nn.functional as F
import horovod.torch as hvd

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
                                         nn.Conv2d(3, 8, 3, padding=1),
                                         nn.BatchNorm2d(8),
                                         nn.ReLU(),
                                         nn.MaxPool2d(2),

                                         nn.Conv2d(8,16, 3, padding=1),
                                         nn.BatchNorm2d(16),
                                         nn.ReLU(),
                                         nn.MaxPool2d(2),

                                         nn.Conv2d(16, 32, 3, padding=1),
                                         nn.BatchNorm2d(32),
                                         nn.ReLU(),
                                         nn.MaxPool2d(2),
                                        )
        self.linear_layers = nn.Sequential(
                                           nn.Linear(512, 100),
                                           nn.BatchNorm1d(100),
                                           nn.ReLU(),
                                           nn.Linear(100,10),
                                           nn.LogSoftmax(dim=1)
                                           )
        torch.nn.init.xavier_normal_(self.conv_layers[0].weight)
        torch.nn.init.xavier_normal_(self.conv_layers[4].weight)
        torch.nn.init.xavier_normal_(self.conv_layers[8].weight)
        torch.nn.init.xavier_normal_(self.linear_layers[0].weight)
        torch.nn.init.xavier_normal_(self.linear_layers[3].weight)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.linear_layers(x) 
        return x



def train(epoch):
    model.train()
    # Horovod: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            # Horovod: use train_sampler to determine the number of examples in
            # this worker's partition.
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_sampler),
                100. * batch_idx / len(train_loader), loss.item()))

def test():
    loss = nn.NLLLoss()
    model.eval()
    test_loss = 0.
    test_accuracy = 0.
    losses = []
    accuracies = []
    for X, y in test_loader:
        prediction = model(X)
        loss_batch = loss(prediction, y)
        losses.append(loss_batch.item())
        prediction = prediction.max(1)[1]
        accuracies.append((prediction==y).float().numpy().mean())
    test_loss = np.mean(losses)
    test_accuracy= np.mean(accuracies)

    print('\nTest set: Loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        test_loss, 100. * test_accuracy))


if __name__ == '__main__':

    hvd.init()
    train_dataset = datasets.CIFAR10('data-%d' % hvd.rank(), train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

    # Horovod: use DistributedSampler to partition the training data.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, sampler=train_sampler)

    #We will validate test only for one root process on whole test dataset
    if hvd.rank() == 0:
        test_dataset = datasets.CIFAR10('data-%d' % hvd.rank(), train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

    # Horovod: use DistributedSampler to partition the test data.
    model = Net()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Horovod: broadcast parameters & optimizer state.
    # So model with root_rank will be averaging of 3 models
    # horovod avarage model parametres
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    #Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         op=hvd.Average)

    if hvd.rank() == 0:
        test()
    for epoch in range(1, 4):
        train(epoch)
        if hvd.rank() == 0:
            test()


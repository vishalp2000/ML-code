from customDataLoader import graspDataSet
from customDataLoader import generate_images

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plot

import time
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib
matplotlib.use('TkAgg', force=True)


class NeuralNetwork(nn.Module):
    def __init__(self, shape, classes=2, nodes=2000):
        super(NeuralNetwork,self).__init__()
        
        self.fc1 = nn.Linear(shape*shape, nodes)
        self.s1  = nn.Sigmoid()
        self.fc2 = nn.Linear(nodes, nodes)
        self.s2  = nn.Sigmoid()
        self.fc3 = nn.Linear(nodes, classes)

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        print(x)
        out = self.fc1(x)
        out = self.s1(out)
        out = self.fc2(out)
        out = self.s2(out)
        out = self.fc3(out)
        return (out)

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # print("X: " + str(X) + "y: " + str(y))

        # Compute prediction error
        pred = model(X)
        #print(f"Pred: {pred}")
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    # print("Accuracy:", round(correct,3), 'Avg loss:', round(test_loss,3))
    return correct, test_loss
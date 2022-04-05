from customDataLoader import graspDataSet
from customDataLoader import generate_images
from model import NeuralNetwork, test, train

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plot

import time
from time import time as now
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib
matplotlib.use('TkAgg', force=True)

import os # For use with the new directory parameter in generateImages

shape = 20

generate_images(shape)

start = now()
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

data_set = graspDataSet(str(shape)+'/labels.csv', str(shape)+'/', ToTensor())
len_data = data_set.__len__()
testl = len_data/2
testt = len_data/2
training_data, testing_data = torch.utils.data.random_split(data_set,[int(testl),int(testt)])

batch_size = int(len_data/2)-1

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)

model = NeuralNetwork(shape).to(device)
#print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

a = []
l = []

while(True):
    try:
        #print('epoch', t)
        train(train_dataloader, model, loss_fn, optimizer, device)
        correct, test_loss = test(test_dataloader, model, loss_fn, device)
        a.append(correct)
        l.append(test_loss)
    except:
        break

plot.plot(a)
plot.plot(l)
plot.show()
print("Done!", correct)


tim = time.localtime()
current_time = time.strftime("%H-%M-%S-", tim)

#torch.save(model.state_dict(), 'models/model.pth')
torch.save(model.state_dict(), 'models/'+str(int(1000*correct))+'_'+current_time+"model.pth")

print('completed training in:', now()-start, 'seconds')

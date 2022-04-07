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
prep_dir = 'C:/Python Files' #'C:/Users/voicu/Downloads/MLCode'

print("Generating Dataset...")
generate_images(prep_dir, shape)

start = now()
# Get cpu or gpu device for training.
device = "cpu" #if torch.cuda.is_available() else "cpu"

print("Fetching Dataset...")
load_dir = os.path.join(prep_dir, 'prepped')
data_set = graspDataSet(os.path.join(load_dir, 'labels.csv'), load_dir+'/', ToTensor())
len_data = data_set.__len__()
# testl = len_data/2
# testt = len_data/2
#replace with sub rather than div by 2
train_set_size = int(len(data_set) * 0.5)
valid_set_size = len(data_set) - train_set_size

training_data, testing_data = torch.utils.data.random_split(data_set,[train_set_size,valid_set_size])

batch_size = int(len_data/2)-1
print('batch_size', batch_size)

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=train_set_size, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=valid_set_size, shuffle=True)

#Display the image to see. 
# import matplotlib.pyplot as plt
# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")
# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")

model = NeuralNetwork(shape).to(device)
# print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

a = []
l = []
x = 0

epochs = 5
correct = 0 
test_loss = 0
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, device=device)
    correct, test_loss = test(test_dataloader, model, loss_fn,device=device)
print("Done!")
# while(True):
#     print("Trying..."+ str(x))
#     try:
#         #print('epoch', t)
#         train(dataloader = train_dataloader, 
#               model = model, 
#               loss_fn = loss_fn, 
#               optimizer = optimizer, 
#               device = device)
#         print('Training is complete?')
#         correct, test_loss = test(test_dataloader, model, loss_fn, device)
#         #print(correct)
#         a.append(correct)
#         l.append(test_loss)
#     except:
#         print("Failed")
#         break
#     x += 1

# # plot.plot(a)
# # plot.plot(l)
# # plot.show()
# print("Done!", correct)

DIR_NAME = 'C:/Users/visha/Downloads/model/models1'

if os.path.isdir(DIR_NAME):
    print(DIR_NAME, "already exists.")
else:
    os.mkdir(DIR_NAME)
    print(DIR_NAME, "is created.")

tim = time.localtime()
current_time = time.strftime("%H-%M-%S-", tim)

#torch.save(model.state_dict(), 'models/model.pth')
torch.save(model.state_dict(), 'models1/'+str(int(1000*correct))+'_'+current_time+"model.pth")

print('completed training in:', now()-start, 'seconds')
cwd = os.getcwd()
print(cwd)
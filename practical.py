import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import os
from PIL import Image

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class network(nn.Module):
    def __init__(self, input_size, out_size):
        super(network, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, out_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x , p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

transform = transforms.Compose(
    [transforms.RandomRotation(20),
     transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))
    ])

class MNISTDataset(Dataset):
    def __init__(self, y=None, transform=None, mode = "train"):
        self.mode = mode
        self.path = os.path.join("data", mode)
        file_name = self.path + '_label.csv'
        self.y = pd.read_csv(file_name, index_col=0, names = ["data", "label"])
        self.transform = transform
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        image = Image.open(os.path.join(self.path, self.mode) + "_" + str(index) + ".jpg")
        if self.transform is not None :
            image = self.transform(image)
        a = self.y.iloc[index][0]
            
        return image, a

model = network(28*28, 10)

EPOCH = 10
BATCH_SIZE = 50

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

trainDataset = MNISTDataset(mode = "train", transform = transform)
train_loader = DataLoader(dataset = trainDataset,
                          batch_size = BATCH_SIZE,
                          shuffle = True,
                          num_workers = 4)

validDataset = MNISTDataset(mode = "valid", transform = transform)
valid_loader = DataLoader(dataset = validDataset,
                          batch_size = BATCH_SIZE,
                          shuffle = False,
                          num_workers = 4)

for i in range(0, EPOCH) :
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = Variable(data.view(-1, 28*28))
        output = model(data)
        loss = loss_fn(output, target.long())
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                i, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))



correct = 0
total = 0
model.eval()
i = 0
for images_, labels in valid_loader:
    images = Variable(images_)
    outputs = model(images.view(-1, 28*28))
    _, predict = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predict == labels).sum()
    i += BATCH_SIZE

print('Accuracy of the network on the valid images: %f %%' % (100.0 * correct.item() / total))

class testDataset(Dataset):
    def __init__(self, transform):
        self.mode = "test"
        self.path = os.path.join("data", self.mode)
        self.transform = transform

    def __len__(self):
        return 10000

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.path, self.mode) + "_" + str(index) + ".jpg")
        image = self.transform(image)
        a = 0

        return image, a

testDataset = testDataset(transform = transforms.Compose([transforms.ToTensor()]))
test_loader = DataLoader(dataset = testDataset,
                          batch_size = BATCH_SIZE,
                          shuffle = False,
                          num_workers = 4)

model.eval()
idx = 0
f = open("submit_file.txt", "w")
f.write("id,label\n");
for images, _ in test_loader:
    images = Variable(images)
    outputs = model(images)
    _, predict = torch.max(outputs.data, 1)
    for i in range(BATCH_SIZE) :
        f.write(str(idx) + "," + str(predict[i].item())+"\n")
        idx += 1
f.close()

torch.save(model.state_dict(), 'model.pkl')

# Define options
class Options():
    pass
opt = Options()

# Training options
opt.batch_size = 50
opt.epochs = 100
opt.learning_rate = 0.01
opt.momentum = 0.9
opt.weight_decay = 5e-4

# Model options
opt.lstm_size = 512

# Backend options
opt.no_cuda = False

#Img options
opt.input_size = 128
opt.sequence_length = 512
opt.num_layers = 1
opt.encoder_layers = 1

'''---------------------------------------------------'''

# Imports
import os, time, torch, sys
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
import torchvision.datasets as Datasets
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data.dataset import Dataset
import numpy as np

globalCounter = 0

# Create datasets   util: https://discuss.pytorch.org/t/questions-about-imagefolder/774/3
# img_short: 200 train, 200 test
# img_train: 80.000 train, 20.000 test

img_dataset = ImageFolder(root='img_short' , transform=ToTensor())
#img_dataset_train =  ImageFolder(root='img_train' , transform=ToTensor())
#img_dataset_test =  ImageFolder(root='img_test' , transform=ToTensor())

class myDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.length = len(datasets)

#come fa ad andare out of range con index?
    def __getitem__(self, index):
        img, label = self.datasets[index]                #PIL.Image(128, 512)
        img = img.mean(0)                                #(128,512)
        img = img.t()                                    #(512, 128)
        img = 1 - img                                    #(512, 128)
        return img, label

    def __len__(self):
        return self.length

def printTrack():
    global globalCounter
    #this just adds a fancy animation
    animationSet = "|/-\\"
    animChar = animationSet[globalCounter % len(animationSet)]
    #end of eyecandy effect

    sys.stdout.write("\r" + animChar + "  Forwarding ")
    sys.stdout.flush()
    globalCounter += 1

img_dataset = myDataset(img_dataset)
# Create loaders
#train_dataset = torch.utils.data.DataLoader(dataset=img_dataset[:80000], batch_size=batch_size, shuffle=True)
#test_dataset = torch.utils.data.DataLoader(dataset=img_dataset[80001:100000], batch_size=batch_size, shuffle=True)
dataset = torch.utils.data.DataLoader(dataset=img_dataset, batch_size=opt.batch_size, shuffle=True)

print("Enumerate dataset: " + str(enumerate(dataset)))

# Define model
class disModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        # Call parent
        super(disModel, self).__init__()
        # Set attributes
        self.is_cuda = False
        self.input_size = input_size
        self.lstm_size = opt.lstm_size
        self.linearOutput = 2                       #binary classification: 0 for dysgraphic, else 1

        # Define modules
        # INPUT IMG -> LSTM -> LINEAR -> SOFTMAX -> OUTPUT 
        self.lstm = nn.LSTM(input_size = self.input_size, hidden_size = self.lstm_size, 
                            num_layers = opt.num_layers, batch_first = True, 
                            dropout = 0, bidirectional = False)
        self.linear = nn.Linear(self.lstm_size,  self.linearOutput)
        
    def cuda(self):
        self.is_cuda = True
        super(disModel, self).cuda()

    def forward(self, x):
        printTrack()            #print state
        # Initial state
        h_0 = Variable(torch.zeros(opt.encoder_layers, opt.input_size, opt.lstm_size))
        c_0 = Variable(torch.zeros(opt.encoder_layers, opt.sequence_length, opt.lstm_size))
        
        # Check CUDA
        #if self.is_cuda:
        #    h_0 = h_0.cuda(async = True)
        #    c_0 = c_0.cuda(async = True)

        # Compute lstm output
        #output, _  = self.lstm(x, (h_0, c_0))              #before custom dataset
        output, _  = self.lstm(x)

        output = self.linear(output[:, -1, :])

        #output = torch.cat(output, 1)       #concat slope and time features just like this from csv
        
        # Compute softmax (commented if train method has his own loss opt)
        '''
        #output = F.log_softmax(output)
        #output = output.view(batch_size, output.size(1), -1)
        '''
        return output

myNN = disModel(opt.input_size, opt.lstm_size, opt.num_layers)
myNN.cuda()      #comment if we are not working with cuda

# Setup loss and optimizier
#criterion = lstm_softmax_loss           #this is custom, taken from daniele's example
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(myNN.parameters(), lr = opt.learning_rate)#, momentum = opt.momentum, weight_decay = opt.weight_decay)

correct = 0
total = 0
# Train the Model
for epoch in range(opt.epochs):
    for i, (images, labels_cpu) in enumerate(dataset):

        images = Variable(images.cuda())
        labels = Variable(labels_cpu.cuda())
        
        optimizer.zero_grad()

        # Forward + Backward + Optimize
        outputs = myNN(images)

        # Compute loss (training only)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)              
        total += labels.size(0)
        correct += (predicted.cpu() == labels_cpu).sum()

        if (i+1) % 5 == 0:
            print ('\rEpoch [%d/%d], Step [%d/%d], Loss: %.4f, Accuracy:  %d %%' 
                    %(epoch+1, opt.epochs, i+1, len(img_dataset)//opt.batch_size, loss.data[0], 100 * correct / total))

# Test the Model
correct = 0
total = 0
for images, labels in test_dataset:
    images = Variable(images.cuda())
    outputs = myNN(images)

    loss = criterion(outputs, labels)

    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels_cpu).sum()

    print('Test Accuracy of the model on the test images: %d %%' % (100 * correct / total)) 

# Save the Model
savingFile = "myNN.pkl"
print("Saving model as " + savingFile)
torch.save(myNN.state_dict(), savingFile)
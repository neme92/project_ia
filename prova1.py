#Define options
class Options(object):
    pass
opt = Options()

#Training options
opt.batch_size = 50
opt.learning_rate = 0.01
opt.epochs = 100
# Backend options
opt.cuda = False

#Imports
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

# Load dataset
#data = ImageFolder(root='img', transform=ToTensor())
#       split in:
#train_dataset = ImageFolder(root='img_train', transform=ToTensor())
#test_dataset = ImageFolder(root='img_test', transform=ToTensor())

img_dataset =  ImageFolder(root='img', transform=ToTensor())
# Create loaders
loaders = {'train': DataLoader(img_dataset[:80000], batch_size = opt.batch_size, shuffle = True,  num_workers = opt.data_workers, pin_memory = False if opt.no_cuda else True),
           'test':  DataLoader(img_dataset[80001:100000],  batch_size = opt.batch_size, shuffle = False, num_workers = opt.data_workers, pin_memory = False if opt.no_cuda else True)}


#for istance, if i'm going to split 80 train - 20 test
num_train = len(train_dataset)
num_test = len(test_dataset)

num_train_batches = math.floor(num_train/opt.batch_size)
num_test_batches = math.floor(num_test/opt.batch_size)

# Convert images to features
train_features = []
train_labels = []
test_features = []
test_labels = []
pil_to_tensor = torchvision.transforms.ToTensor()
for img in train_dataset:
    #no need to standardize or scale the imgs, already done while collecting them
    if opt.cuda: img = img.cuda()
    # Forward
    features = model(Variable(img.unsqueeze(0))).data.cpu().clone()
    # Add to lists
    train_features.append(features)
    train_labels.append(label)
    print("\rProcessing training set: " + len(train_features) + "/" + num_train + " " * 20)
print()
for (img, label) in test_dataset:
    # Scale image
    img = scale(img)
    # Standardize image
    img = (pil_to_tensor(img) - 0.5)/0.25
    if opt.cuda: img = img.cuda()
    # Forward
    features = model(Variable(img.unsqueeze(0))).data.cpu().clone()
    # Add to lists
    test_features.append(features)
    print("\rProcessing training set: " + len(train_features) + "/" + num_train + " " * 20)
print()
# Get features size
feat_size = train_features[0].size(1)

# Define model
class Model(nn.Module):
    def __init__(self):
        # Call parent
        super(Model, self).__init__()
        # Set attributes, if any

        # Define modules
        # INPUT IMG -> LSTM -> LINEAR -> SOFTMAX -> OUTPUT 

        self.lstm = nn.LSTM(seq_len = 512, input_size = 128, batch_first = True, bidirectional = False)
        self.linear = nn.Linear(feat_size, 2)

    def forward(self, input):
        # Compute output
        input_seq = input #from input, get tensors from img
        #input_seq = input_seq.unsqueeze(0) # Adds 1 dim for batch, needed even if already unsqueezed while passing the img as arg to forward?
        output_seq = self.lstm(input_seq, )

        #select last state
        last_output = output_seq[-1]

        #last output should hold last state of the lstm
        # here must torch.cat with angle and time data and then 
        # this array must be passed to linear layer to get the final output
        linear_output = self.linear(last_output)        #returns (batch, 2)

        # Compute softmax
        res = F.log_softmax(linear_output)  
        return res

model = Model()
if opt.cuda: model.cuda()

# Define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr = opt.learning_rate)

# Setup loss criterion
criterion = nn.NLLLoss()
if opt.cuda: criterion.cuda()

# Start training
for epoch in range(1, opt.epochs+1):
    # Training mode
    model.train()
    # Shuffle training data indexes
    shuffled_idx = torch.randperm(num_train_batches)
    # Initialize training loss and accuracy
    train_loss = 0
    train_accuracy = 0
    
    # Process all training batches
    for i in range(0, num_train_batches):
        # Prepare batch
        batch_start = i*opt.batch_size
        batch_idx = shuffled_idx[batch_start:batch_start+opt.batch_size]
        input = torch.Tensor(opt.batch_size, feat_size)
        target = torch.LongTensor(opt.batch_size)
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
        # Fill batch
        for j in range(0, batch_idx.numel()):
            # Copy to batch
            input[j] = train_features[batch_idx[j]]
            target[j] = train_labels[batch_idx[j]]
        # Wrap for autograd
        input = Variable(input)
        target = Variable(target)
        # Forward
        output = model(input)
        loss = criterion(output, target)
        train_loss += loss.data[0]
        # Compute accuracy
        _,pred = output.data.max(1)
        correct = pred.eq(target.data).sum()
        accuracy = correct/opt.batch_size
        train_accuracy += accuracy
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Evaluation mode
    model.eval()
    # Initialize test loss and accuracy
    test_loss = 0
    test_accuracy = 0
    # Process all test batches
    for i in range(0, num_test_batches):
        # Prepare batch
        batch_start = i*opt.batch_size
        batch_idx = list(range(batch_start, batch_start+opt.batch_size))
        input = torch.Tensor(opt.batch_size, feat_size)
        target = torch.LongTensor(opt.batch_size)
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
        # Fill batch
        for j in range(0, len(batch_idx)):
            # Copy to batch
            input[j] = test_features[batch_idx[j]]
            target[j] = test_labels[batch_idx[j]]
        # Wrap for autograd
        input = Variable(input, volatile=True)
        target = Variable(target, volatile=True)
        # Forward
        output = model(input)
        loss = criterion(output, target)
        test_loss += loss.data[0]
        # Compute accuracy
        _,pred = output.data.max(1)
        correct = pred.eq(target.data).sum()
        accuracy = correct/opt.batch_size
        test_accuracy += accuracy
    # Print loss/accuracy
        print("Epoch {0}: TrL={1:.4f}, TrA={2:.4f}, TeL={3:.4f}, TeA={4:.4f}".format(epoch,
                                                                                 train_loss/num_train_batches,
                                                                                 train_accuracy/num_train_batches,
                                                                                 test_loss/num_test_batches,
                                                                                 test_accuracy/num_test_batches))
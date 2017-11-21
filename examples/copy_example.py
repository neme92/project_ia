# Define options
class Options():
    pass
opt = Options()
# Training options
opt.batch_size = 64
opt.learning_rate = 0.01
opt.learning_rate_decay_by = 0.8
opt.learning_rate_decay_every = 10
opt.weight_decay = 5e-4
opt.momentum = 0.9
opt.data_workers = 0 # load on main thread
opt.epochs = 1000
# Checkpoint options
opt.save_every = 20
# Model options
opt.encoder_layers = 1
opt.lstm_size = 1024
# Test options
import sys
opt.model = sys.argv[1] if len(sys.argv) > 1 else None
opt.test = sys.argv[2] if len(sys.argv) > 2 else None
# Backend options
opt.no_cuda = False

# Imports
import os
import time
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
from dataset import Dataset

# Create datasets
train_dataset = Dataset(source_file = "data/train/sources.txt", target_file = "data/train/targets.txt")
test_dataset = Dataset(source_file = "data/test/sources.txt", target_file = "data/test/targets.txt")
# Create loaders
loaders = {'train': DataLoader(train_dataset, batch_size = opt.batch_size, shuffle = True,  num_workers = opt.data_workers, pin_memory = False if opt.no_cuda else True),
           'test':  DataLoader(test_dataset,  batch_size = opt.batch_size, shuffle = False, num_workers = opt.data_workers, pin_memory = False if opt.no_cuda else True)}

# Define model
class Model1(nn.Module):

    def __init__(self, input_size, sos_idx, eos_idx, encoder_layers = 1, lstm_size = 128):
        # Call parent
        super(Model1, self).__init__()
        # Set attributes
        self.is_cuda = False
        self.input_size = input_size
        self.encoder_layers = encoder_layers
        self.lstm_size = lstm_size
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        # Define modules
        self.encoder = nn.LSTM(input_size = self.input_size, hidden_size = self.lstm_size, num_layers = encoder_layers, batch_first = True, dropout = 0, bidirectional = False)
        self.decoder = nn.LSTM(input_size = self.input_size, hidden_size = self.lstm_size, num_layers = 1,              batch_first = True, dropout = 0, bidirectional = False)
        self.dec_to_output = nn.Linear(self.lstm_size, self.input_size)
        
    def cuda(self):
        self.is_cuda = True
        super(Model1, self).cuda()

    def forward(self, x, target_as_input = None):
        # Get input info
        batch_size = x.size(0)
        seq_len = x.size(1)
        # Initial state
        h_0 = Variable(torch.zeros(self.encoder_layers, batch_size, self.lstm_size))
        c_0 = Variable(torch.zeros(self.encoder_layers, batch_size, self.lstm_size))
        # Check CUDA
        if self.is_cuda:
            h_0 = h_0.cuda(async = True)
            c_0 = c_0.cuda(async = True)
        # Compute encoder output (hidden layer at last time step)
        x = self.encoder(x, (h_0, c_0))[0][:,-1,:]
        # Prepare decoder state
        h_0 = x.unsqueeze(0) # Adds num_layers dimension
        c_0 = Variable(torch.zeros(h_0.size()))
        # Check CUDA
        if self.is_cuda:
            c_0 = c_0.cuda(async = True)
        # If target_as_input is provided (during training), target is input sequence; otherwise output is fed back as input
        if target_as_input is not None:
            # Compute decoder output
            x = self.decoder(target_as_input, (h_0, c_0))[0].contiguous()
            # Compute outputs
            #print(x.data.size())
            x = x.view(-1, self.lstm_size)
            #print(x.data.size())
            x = self.dec_to_output(x)
            #print(x.data.size())
            # Compute softmax
            x = F.log_softmax(x)
            #print(x.data.size())
            x = x.view(batch_size, target_as_input.size(1), -1)
        else:
            # Initialize input
            input = torch.zeros(batch_size, 1, self.input_size)
            input[:, :, self.sos_idx].fill_(1)
            input = Variable(input)
            if self.is_cuda:
                input = input.cuda()
            h = h_0
            c = c_0
            # Initialize list of outputs at each time step
            output = []
            # Process until EOS is found or limit is reached
            for i in range(0, 50): # TODO this must be dependent on dataset
                # Get decoder output at this time step
                o, hc = self.decoder(input, (h, c))
                h, c = hc
                # Compute output
                o2 = self.dec_to_output(o.view(-1, self.lstm_size))
                # Compute log-softmax
                o2 = F.log_softmax(o2)
                # View as sequence and add to outputs
                o2 = o2.view(batch_size, 1, -1)
                output.append(o2)
                # Compute predicted outputs
                output_idx = o2.data.max(2)[1].squeeze()
                # Check all words are in EOS
                if (output_idx == self.eos_idx).all():
                    break
                # Compute input for next step
                input = torch.zeros(batch_size, 1, self.input_size)
                for j in range(0, batch_size):
                    input[j, 0, output_idx[j]] = 1
                input = Variable(input)
                if self.is_cuda:
                    input = input.cuda(async = True)
            # Concatenate all log-softmax outputs
            x = torch.cat(output, 1)
        return x
    
def lstm_softmax_loss(output, target):
    # Merge sequence dimension and batch dimension
    batch_size = output.size(0)
    seq_size = output.size(1)
    output = output.view(batch_size*seq_size, -1)
    target = target.contiguous().view(batch_size*seq_size, -1)
    # Convert targets to "class" indeces
    _,target = target.max(1)
    target = target.squeeze()
    # Compute standard NLL loss
    loss = F.nll_loss(output, target)
    return loss


# Check test
if opt.test is None:
    # Create model/optimizer for training
    model_options = {'input_size': train_dataset[0][0].size(1), 'sos_idx': train_dataset.sos_idx, 'eos_idx': train_dataset.eos_idx, 'encoder_layers': opt.encoder_layers, 'lstm_size': opt.lstm_size}
    model = Model1(**model_options)
    optimizer = torch.optim.SGD(model.parameters(), lr = opt.learning_rate, momentum = opt.momentum, weight_decay = opt.weight_decay)
else:
    # Load checkpoint
    checkpoint = torch.load(opt.model)
    # Load parameters
    model_options = checkpoint["model_options"]
    # Create model
    model = Model1(**model_options)
    model.load_state_dict(checkpoint["model_state"])
    # Define input sequence
    input = [int(x) for x in opt.test.split(" ")]
    # Convert to one-hot
    input_onehot = torch.zeros(1, train_dataset.seq_len, train_dataset.vocab_len)
    input_onehot[0, 0, train_dataset.sos_idx] = 1
    for i in range(1, train_dataset.seq_len):
        if i <= len(input):
            input_onehot[0, i, input[i-1]] = 1
        else:
            input_onehot[0, i, train_dataset.eos_idx] = 1
    input = input_onehot
    # Feed to model
    model.cuda()
    model.eval()
    output = model(Variable(input.cuda(), volatile = True))
    # Convert to symbols
    _,output = output.data.max(2)
    output = output.squeeze()
    print(output.tolist()[:-1])
    sys.exit(0)
    
# Setup loss criterion
criterion = lstm_softmax_loss

# Setup CUDA
if not opt.no_cuda:
    model.cuda()

# Monitoring options
update_every = 100
cnt = 0
# Start training
try:
    for epoch in range(1, opt.epochs+1):
        # Adjust learning rate for SGD
        lr = opt.learning_rate * (opt.learning_rate_decay_by ** (epoch // opt.learning_rate_decay_every))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # Initialize loss/accuracy
        sum_loss = 0 # training only
        n_train = 0 # for averaging
        sum_length_accuracy = 0 # test only
        sum_match_accuracy = 0 # test only
        n_test = 0 # for averaging
        for split in ["train", "test"]:
            # Set mode
            if split == "train":
                model.train()
            else:
                model.eval()
            # Process all training batches
            for i, (input, target) in enumerate(loaders[split]):
                # Check CUDA
                if not opt.no_cuda:
                    input = input.cuda(async = True)
                    target = target.cuda(async = True)
                # Wrap for autograd
                input = Variable(input, volatile = (split != "train"))
                target_as_input = Variable(target[:, :-1, :], volatile = (split != "train"))
                target_as_target = Variable(target[:, 1:, :], volatile = (split != "train"))
                # Forward (use target as decoder input for training)
                output = model(input, target_as_input if split == "train" else None)
                # Compute loss (training only)
                loss = criterion(output, target_as_target) if split == "train" else Variable(torch.Tensor([-1]))
                sum_loss += loss.data[0]
                n_train += 1
                # Compute accuracy
                if split != "train": # and epoch > 10:
                    # Get one-hot indices of output and target
                    _,output_idx = output.data.max(2)
                    output_idx = output_idx.squeeze().cpu()
                    _,target_idx = target_as_target.data.max(2)
                    target_idx = target_idx.squeeze().cpu()
                    # Compute length of each sequence
                    _,output_lengths = (output_idx == train_dataset.eos_idx).max(1)
                    _,target_lengths = (target_idx == train_dataset.eos_idx).max(1)
                    # Compute length accuracy
                    length_accuracy = torch.mean(output_lengths.float()/target_lengths.float())
                    # Compute matching lengths
                    match_lengths = torch.min(output_lengths, target_lengths)
                    # Compute matching degrees
                    if match_lengths.max() == 0:
                        match_degrees = torch.zeros(match_lengths.nelement(), 1)
                    else:
                        max_common_length = match_lengths.max()
                        output_idx = output_idx[:, :max_common_length]
                        target_idx = target_idx[:, :max_common_length]
                        match_degrees = (output_idx == target_idx).cumsum(1).float()/(match_lengths.float().expand_as(output_idx))
                    # Fix infinity (when length is 0)
                    match_lengths[match_lengths == 0] = 1
                    match_degrees[match_degrees == float('inf')] = 0
                    # Get match degrees at corresponding lengths
                    match_degrees = match_degrees.view(-1)[(torch.arange(0, match_degrees.size(0))*match_degrees.size(1)).long() + match_lengths.long() - 1]
                    # Compute match accuracy
                    match_accuracy = match_degrees.mean()
                    # Update monitoring variables
                    sum_length_accuracy += length_accuracy
                    sum_match_accuracy += match_accuracy
                    n_test += 1
                # Backward and optimize
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                # Show results every once in a while
                cnt += 1
                if cnt % update_every == 0:
                    print("Epoch {0}: L = {1:.4f}, LA = {2:.4f}, MA = {3:.4f}\r".format(epoch, sum_loss/n_train, sum_length_accuracy/n_test if n_test > 0 else -1, sum_match_accuracy/n_test if n_test > 0 else -1), end = '')
        # Print info at the end of the epoch
        print("Epoch {0}: L = {1:.4f}, LA = {2:.4f}, MA = {3:.4f}".format(epoch, sum_loss/n_train, sum_length_accuracy/n_test if n_test > 0 else -1, sum_match_accuracy/n_test if n_test > 0 else -1))
        # Save checkpoint
        if epoch % opt.save_every == 0:
            # Build file name
            checkpoint_path = "checkpoint-" + repr(epoch) + ".pth"
            # Write data
            torch.save({'model_options': model_options, 'model_state': model.state_dict()}, checkpoint_path)
except KeyboardInterrupt:
    print("Interrupted")

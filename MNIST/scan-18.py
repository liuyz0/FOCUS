# we want to scan different optimizers and learning rates
# and different batch sizes
# Adam vs Signum vs FOCUS
# 12 dfferent batch sizes

import sys
import torch
from torch import nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time
from focus import FOCUS

# Grab the argument that is passed in
# This is the index into fnames for this process
task_id = int(sys.argv[1]) # from 0 to 48
num_tasks = int(sys.argv[2]) # should be 12 * 4 = 48
num_replicates = 3

opt_idx = task_id // 12
bs_idx = task_id % 12

opt_names = ['Adam', 'FOCUS0', 'FOCUS2', 'FOCUS4']
lr_ran = 10 ** torch.linspace(-4, 0, 20)
weight_decay = 1e-2

opt_name = opt_names[opt_idx]

# define the loss functions
loss_func=nn.CrossEntropyLoss()

## load data
bs_span = 2**torch.arange(1, 13)
batch_size = bs_span[bs_idx].item()
transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(size=28, antialias=None)
            ])
train_data = datasets.MNIST(root = '../../Selfprune/data', train = True,
                        transform = transform, download = False)
test_data = datasets.MNIST(root = '../../Selfprune/data', train = False,
                        transform = transform, download = False)
num_classes= len(train_data.classes)

train_loader = torch.utils.data.DataLoader(dataset = train_data,
                                            batch_size = batch_size,
                                            shuffle = True,
                                            pin_memory=True)
test_loader = torch.utils.data.DataLoader(dataset = test_data,
                                            batch_size = 8192,
                                            shuffle = False,
                                            pin_memory=True)

# Define a Multi-Layer Perceptron (MLP) class
class MLP(nn.Module):
    # Initialize the MLP with a list of layer widths and an optional dropout probability
    def __init__(self, width_list, p=0, use_batch_norm=True):
        # Call the parent class's constructor
        super(MLP, self).__init__()
        self.use_batch_norm = use_batch_norm

        # Calculate the depth of the network
        self.depth = len(width_list) - 1

        # Initialize lists for fully connected (fc) layers and batch normalization (bn) layers
        self.fc_list = torch.nn.ModuleList([])
        self.bn_list = torch.nn.ModuleList([])

        # Set the dropout probability
        self.p = p

        # Create the fc and bn layers based on the width list
        for i in range(self.depth):
            self.fc_list.append(nn.Linear(width_list[i], width_list[i + 1]))
            if use_batch_norm:
                self.bn_list.append(nn.BatchNorm1d(width_list[i]))

        # If dropout is enabled, initialize dropout (do) layers
        if self.p > 0:
            self.do_list = torch.nn.ModuleList([])
            for i in range(self.depth - 1):
                self.do_list.append(nn.Dropout(p=p))

    # Define the forward pass of the MLP
    def forward(self, x):
        
        # Perform the forward pass through each layer except the last
        for i in range(self.depth - 1):
            # Apply fc and bn layers, followed by ReLU activation
            if self.use_batch_norm:
                x = self.bn_list[i](x)
            x = self.fc_list[i](x)
            x = x.relu()

            # Apply dropout if enabled
            if self.p > 0:
                x = self.do_list[i](x)

        # Perform the forward pass through the last layer without ReLU activation
        if self.use_batch_norm:
            x = self.bn_list[-1](x)
        x = self.fc_list[-1](x)

        # Return output of the last layer
        return x
    

results = torch.zeros(20, num_replicates, 3, 400)
def create_optimizer(model, learning_rate, weight_decay, opt_idx, gamma=0.0):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if any(nd in name for nd in ['bias', 'bn']):
            no_decay.append(param)
        else:
            decay.append(param)
    if opt_idx == 0:
        return torch.optim.AdamW(
            [
                {'params': no_decay, 'weight_decay': 0.0},
                {'params': decay, 'weight_decay': weight_decay}
            ],
            lr=learning_rate
        )
    else:
        return FOCUS(
            [
                {'params': no_decay, 'weight_decay': 0.0},
                {'params': decay, 'weight_decay': weight_decay}
            ],
            lr=learning_rate,
            betas=(0.9, 0.99),
            gamma=gamma
        )

start_time = time.time()

for lr_idx in range(20):
    lr = lr_ran[lr_idx]
    for replicate in range(num_replicates):
        model = MLP([784, 128, 128, 128, 128, 128, num_classes], p=0)
        if opt_idx == 0:
            optimizer = create_optimizer(model, lr, weight_decay, opt_idx)
        elif opt_idx == 1:
            optimizer = create_optimizer(model, lr, weight_decay, opt_idx, 0.0)
        elif opt_idx == 2:
            optimizer = create_optimizer(model, lr, weight_decay, opt_idx, 0.2)
        elif opt_idx == 3:
            optimizer = create_optimizer(model, lr, weight_decay, opt_idx, 0.4)
        
        train_loss = []
        test_loss = []
        test_acc = []
        step = 0

        while True:
            for data in train_loader:
                model.train()
                img, label = data
                img = img.view(-1,784)
                optimizer.zero_grad()
                output = model(img)
                loss = loss_func(output,label)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
            
                model.eval()
                batch_loss = 0
                num_data = 0
                correctnum = 0
                with torch.no_grad():
                    for data in test_loader:
                        img, label = data
                        img = img.view(-1,784)
                        output = model(img)
                        loss = loss_func(output,label)
                        batch_loss += loss.item()*label.numel()
                        num_data += label.numel()
                        correctnum += torch.eq(torch.argmax(output, dim=-1), label).float().sum().item()
                test_loss.append(batch_loss / num_data)
                test_acc.append(correctnum / num_data)
                step += 1

                if step == 400:
                    break
            if step == 400:
                break
        print(f'opt = {opt_name}, lr={lr}, replicate={replicate}, test_acc={test_acc[-1]}')

        # cat the results
        results[lr_idx,replicate, 0, :] = torch.tensor(train_loss)
        results[lr_idx,replicate, 1, :] = torch.tensor(test_loss)
        results[lr_idx,replicate, 2, :] = torch.tensor(test_acc)

# save the results
torch.save(results, f'../outputs/scan-18-{task_id}.pt')
print(f'Total time: {time.time() - start_time}')
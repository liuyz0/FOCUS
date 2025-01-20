# 96 weight decays by 100 noise std
# each pixcel 3 optimizers
# each optimizer 20 lr
# 50 replicates and each replicate run 1000 steps
# change theta to pi/6

import numpy as np
import sys
import time

# define the class
class Runexp(object):
    def __init__(self, p_landscapes = (1,1), lr = 0.01, optimizer = 'Adam', 
                 betas = (0.9, 0.999), gamma = 0.2, epsilon = 1e-8, theta = 0, weight_decay = .1):
        self.a, self.c = p_landscapes
        self.lr = lr
        self.p = np.random.randn(2) * 1e-4
        self.optimizer = optimizer
        self.exp_avg = np.zeros(2)
        self.exp_avg_sq = np.zeros(2)
        self.pbar = np.zeros(2)
        self.t = 0
        self.beta1, self.beta2 = betas
        self.gamma = gamma
        self.epsilon = epsilon
        self.matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        self.weight_decay = weight_decay

    def loss(self):
        # p is a 1D array with 2 elements
        cp = self.p @ self.matrix
        return 0.5 * self.a* cp[0]**2 * cp[1]**2 - self.c * cp[0]
    
    def grad_loss(self, std = 0, mean = 1.0):
        cp = self.p @ self.matrix
        g1 = self.a* cp[0] * cp[1]**2 - self.c
        g2 = self.a* cp[0]**2 * cp[1]
        return (np.array([g1, g2]) @ self.matrix.T) * np.random.normal(mean, std, 2)

    def step(self, std = 0, mean = 1.0):
        self.t += 1
        g = self.grad_loss(std = std, mean = mean)
        if self.optimizer == 'Adam':
            self.exp_avg = self.beta1 * self.exp_avg + (1 - self.beta1) * g
            self.exp_avg_sq = self.beta2 * self.exp_avg_sq + (1 - self.beta2) * g**2
            exp_avg_bar = self.exp_avg / (1 - self.beta1**self.t)
            exp_avg_sq_bar = self.exp_avg_sq / (1 - self.beta2**self.t)
            self.p -= self.lr * (exp_avg_bar / (np.sqrt(exp_avg_sq_bar) + self.epsilon) + self.weight_decay * self.p)
        elif self.optimizer == 'FOCUS':
            self.exp_avg = self.beta1 * self.exp_avg + (1 - self.beta1) * g
            self.pbar = self.beta2 * self.pbar + (1 - self.beta2) * self.p
            exp_avg_bar = self.exp_avg / (1 - self.beta1**self.t)
            pbar_hat = self.pbar / (1 - self.beta2**self.t)
            self.p -= self.lr * (np.sign(exp_avg_bar) + self.gamma * np.sign(self.p - pbar_hat) + self.weight_decay * self.p)
        else:
            raise ValueError('Unknown optimizer')
        return self.p

# define parameters
# Grab the argument that is passed in
# This is the index into fnames for this process
task_id = int(sys.argv[1]) # from 0 to 95
num_tasks = int(sys.argv[2]) # should be 96

weight_decays = np.linspace(0,0.5,96)
weight_decay = weight_decays[task_id]

stds = np.linspace(0, 3, 100)
lrs = np.logspace(-3, 0, 20)
losses = np.zeros((3, 100, 20, 50, 3))
# 0 is final loss
# 1 is historically minimal loss
losses[:,:,:,:,1] = np.inf
# 2 is projection to valley axis

# Run it

start_time = time.time()
for optim_idx in range(3):
    gamma = 0.2
    if optim_idx == 0:
        optimizer = 'Adam'
        beta2 = 0.999
    else:
        optimizer = 'FOCUS'
        beta2 = 0.9
        if optim_idx == 2:
            gamma = 0.0
            beta2 = 0
    for std_idx in range(100):
        std = stds[std_idx]
        for lr_idx in range(20):
            lr = lrs[lr_idx]
            for replicate in range(50):
                run_optim = Runexp(p_landscapes = (10,.1), lr = lr, optimizer = optimizer, betas = (0.9, beta2), 
                                   gamma = gamma, epsilon = 1e-8, theta = np.pi/6, weight_decay=weight_decay)
                for step in range(1000):
                    run_optim.step(std = std)
                    run_loss = run_optim.loss()
                    valley_dim = np.dot(run_optim.p, np.array([np.cos(np.pi/6), np.sin(np.pi/6)]))
                    if run_loss < losses[optim_idx,std_idx,lr_idx,replicate,1]:
                        losses[optim_idx,std_idx,lr_idx,replicate,1] = run_loss
                    if valley_dim > losses[optim_idx,std_idx,lr_idx,replicate,2]:
                        losses[optim_idx,std_idx,lr_idx,replicate,2] = valley_dim
                losses[optim_idx,std_idx,lr_idx,replicate,0] = run_optim.loss()
    
        print(f'To optimizer: {optimizer}, std_idx = {std_idx}')

print(f'Total time: {time.time() - start_time}')
np.save(f'../outputs/syn-exp-6-{task_id}.npy', losses)

wandb_log = True
wandb_project = 'FOCUS'
wandb_run_name='gpt2-small-signGD-50k'

# these make the total batch size be ~0.5M
# 8 batch size * 1024 block size * 6 gradaccum * 10 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False

# this makes total number of tokens be 300B
max_iters = 50000
lr_decay_iters = 100000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# optimizer
optimizer_name = 'focus'
learning_rate = 1e-4
rho = 0.0 # gamma
weight_decay = 2e-1 # increase to 0.2 later if nan
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
min_lr = 1.5e-5 

#init_from = 'resume'

compile = True

out_dir = 'out_small_signGD_50k'
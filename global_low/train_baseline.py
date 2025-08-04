# train_baseline.py
# This script is used to train baseline models.

import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import torch.utils.data as data_utils
import yaml

with open('config_baseline.yaml', 'r') as f:
    config = yaml.safe_load(f)

selected_model = config['selected_model']

model_config = config['models'][selected_model]
training_config = config['trainings'][selected_model]
data_config = config['datas'][selected_model]
logging_config = config['loggings'][selected_model]

backbone = logging_config['backbone']
log_dir = logging_config['log_dir']
checkpoint_dir = logging_config['checkpoint_dir']
result_dir = logging_config['result_dir']

os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

logging.basicConfig(filename=f'{log_dir}/{backbone}_training_log.log',
                    level=logging.INFO,
                    format='%(asctime)s %(message)s')

seed = training_config['seed']

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(seed)

# ============== Distributed Training Settings ===============
parallel_method = training_config.get('parallel_method', 'DistributedDataParallel')

if parallel_method == 'DistributedDataParallel':
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    num_gpus = dist.get_world_size()

    def reduce_mean(tensor, nprocs):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= nprocs
        return rt
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    local_rank = 0  # For DataParallel, set to 0

    def reduce_mean(tensor, nprocs):
        return tensor

# ============== Data loader ==============
from dataloader_baseline import train_Dataset, test_Dataset

train_dataset = train_Dataset(data_config['data_path'])
val_dataset = test_Dataset(data_config['data_path'])

if parallel_method == 'DistributedDataParallel':
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
else:
    train_sampler = None
    val_sampler = None

train_loader = data_utils.DataLoader(
    train_dataset,
    num_workers=1,  # Adjust as needed
    batch_size=training_config['batch_size'],
    sampler=train_sampler,
    pin_memory=True
)

val_loader = data_utils.DataLoader(
    val_dataset,
    num_workers=1,
    batch_size=training_config['batch_size'],
    sampler=val_sampler,
    shuffle=False,
    pin_memory=True
)

# ============== Model settings ==============
# Model registry
from model_baselines.fno import FNO2d 
from model_baselines.dit import Dit
from model_baselines.simvp import SimVP
from model_baselines.cno import CNO
from model_baselines.pastnet import PastNetModel
from model_baselines.resnet import ResNet
from model_baselines.unet import U_net
from model_baselines.convlstm import ConvLSTM_NS
from model_baselines.lsm import LSM
# Add more imports as needed, e.g.
# from baseline.UNet import UNet

model_dict = {
    'FNO': FNO2d,
    'Dit': Dit,
    'SimVP': SimVP,
    'CNO': CNO,
    'PastNet': PastNetModel,
    'ResNet': ResNet,
    'U_net': U_net,
    'ConvLSTM': ConvLSTM_NS,
    'LSM': LSM
}

model_name = selected_model
if model_name in model_dict:
    ModelClass = model_dict[model_name]
    model = ModelClass(**model_config['parameters'])
else:
    raise ValueError(f"Model {model_name} is not defined.")

model = model.to(device)

# Process according to the parallelization method
if parallel_method == 'DistributedDataParallel':
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
elif parallel_method == 'DataParallel':
    model = nn.DataParallel(model)
else:
    raise ValueError(f"Unknown parallel method: {parallel_method}")

# ============== Loss Function and Optimizer ==============
criterion = nn.MSELoss()
learning_rate = training_config['learning_rate']
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(
    optimizer, 
    step_size=training_config['lr_step_size'],
    gamma=training_config['lr_gamma']
)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, training_config['lr_step_size'], eta_min=0)

# ============== Training, validation, and testing functions ==============
def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    if parallel_method == 'DistributedDataParallel' and train_loader.sampler is not None:
        train_loader.sampler.set_epoch(epoch)
    train_loss = 0.0
    for input_vel, target_vel in tqdm(train_loader, desc="Training", disable=local_rank != 0):
        input_vel = input_vel.to(device, non_blocking=True).float()
        target_vel = target_vel.to(device, non_blocking=True).float()
        optimizer.zero_grad()
        output_vel = model(input_vel)
        loss = criterion(output_vel, target_vel)
        loss.backward()
        optimizer.step()
        loss_value = reduce_mean(loss, num_gpus).item() if parallel_method == 'DistributedDataParallel' else loss.item()
        train_loss += loss_value * input_vel.size(0)
    return train_loss / len(train_loader.dataset)

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for input_vel, target_vel in tqdm(val_loader, desc="Validation", disable=local_rank != 0):
            input_vel = input_vel.to(device, non_blocking=True).float()
            target_vel = target_vel.to(device, non_blocking=True).float()
            output_vel = model(input_vel)
            loss = criterion(output_vel, target_vel)
            loss_value = reduce_mean(loss, num_gpus).item() if parallel_method == 'DistributedDataParallel' else loss.item()
            val_loss += loss_value * input_vel.size(0)
    return val_loss / len(val_loader.dataset)

def test(model, val_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    all_inputs = []
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for input_vel, target_vel in tqdm(val_loader, desc="Testing", disable=local_rank != 0):
            input_vel = input_vel.to(device, non_blocking=True).float()
            target_vel = target_vel.to(device, non_blocking=True).float()
            output_vel = model(input_vel)

            if local_rank == 0:
                all_inputs.append(input_vel.cpu().numpy())
                all_targets.append(target_vel.cpu().numpy())
                all_outputs.append(output_vel.cpu().numpy())

            loss = criterion(output_vel, target_vel)
            loss_value = reduce_mean(loss, num_gpus).item() if parallel_method == 'DistributedDataParallel' else loss.item()
            test_loss += loss_value * input_vel.size(0)

    if local_rank == 0:
        all_inputs = np.concatenate(all_inputs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_outputs = np.concatenate(all_outputs, axis=0)

        np.save(f'{result_dir}/{backbone}_inputs.npy', all_inputs)
        np.save(f'{result_dir}/{backbone}_targets.npy', all_targets)
        np.save(f'{result_dir}/{backbone}_outputs.npy', all_outputs)

    return test_loss / len(val_loader.dataset)

# ============== Load Checkpoint if exists ==============
best_model_path = f'{checkpoint_dir}/{backbone}_best_model.pth'

if local_rank == 0 and os.path.exists(best_model_path):
    try:
        logging.info('Loading best model from checkpoint.')
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint)
    except Exception as e:
        logging.error(f'Error loading model checkpoint: {e}')

# ============== Main training Loop ==============
best_val_loss = float('inf')
num_epochs = training_config['num_epochs']

for epoch in range(num_epochs):
    if local_rank == 0:
        logging.info(f'Epoch {epoch + 1}/{num_epochs}')
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
    val_loss = validate(model, val_loader, criterion, device)

    scheduler.step()

    if local_rank == 0:
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f'Current Learning Rate: {current_lr:.10f}')
        # torch.save(model.state_dict(), best_model_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        logging.info(f'Train Loss: {train_loss:.7f}, Val Loss: {val_loss:.7f}')

# Optional: Test after training
# if local_rank == 0:
#     model.load_state_dict(torch.load(best_model_path))
#     test_loss = test(model, val_loader, criterion, device)
#     logging.info(f"Testing completed. Test Loss: {test_loss:.7f}")

if parallel_method == 'DistributedDataParallel':
    dist.destroy_process_group()
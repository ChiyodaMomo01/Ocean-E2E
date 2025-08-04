# train.py
# This script is used to train the QGM model components,
# including model_gm and model_S for geostrophic velocity and source term.

import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import yaml

with open('config.yaml', 'r') as f:
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
ckpt_path_pre = logging_config.get('ckpt_path_pre', None)

os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

logging.basicConfig(filename=f'{log_dir}/{backbone}_training_log.log', level=logging.INFO, format='%(asctime)s %(message)s')

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
from dataloader_05res import train_Dataset, test_Dataset

train_dataset = train_Dataset(data_config['data_path'])
val_dataset = test_Dataset(data_config['data_path'])

if parallel_method == 'DistributedDataParallel':
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
else:
    train_sampler = None
    val_sampler = None

train_loader = DataLoader(
    train_dataset,
    num_workers=training_config['num_workers'],
    batch_size=training_config['batch_size'],
    sampler=train_sampler,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    num_workers=training_config['num_workers'],
    batch_size=training_config['batch_size'],
    sampler=val_sampler,
    shuffle=False,
    pin_memory=True
)

# ============== Model settings ==============
# Model registry
from model_baselines.SimVP import SimVP

class Ocean_e2e(nn.Module):
    def __init__(self, dt, steps):
        """ Initialize the Ocean_e2e model.
        Args:
            dt (float): Time step in seconds
            steps (int): Number of integration steps
        """
        super(Ocean_e2e, self).__init__()
        self.dt = dt
        self.steps = steps
        self.R_earth = 6371e3  # Earth radius in meters
        self.omega = 7.2921e-5  # Earth's angular velocity in rad/s
        self.g = 9.81  # Gravity in m/s²
        self.deg2rad = torch.pi / 180.0

        # Define a fixed 3x3 Gaussian-like kernel for horizontal filtering
        kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
        self.kernel = kernel.view(1, 1, 3, 3)  # Shape for conv2d: (out_channels, in_channels, height, width)

    def apply_horizontal_filter(self, field, mask):
        """ Apply horizontal filter to suppress grid-scale noise using convolution.
        Args:
            field (torch.Tensor): Field to filter, shape (B, H, W)
            mask (torch.Tensor): Mask tensor, shape (B, H, W) or (H, W)
        Returns:
            torch.Tensor: Filtered field, shape (B, H, W)
        """
        # Expand field to (B, 1, H, W) for conv2d
        field_unsq = field.unsqueeze(1)  # Shape (B, 1, H, W)
        kernel = self.kernel.to(field.device).to(field.dtype)
        filtered = F.conv2d(field_unsq, kernel, padding=1)  # 'same' padding
        return filtered.squeeze(1) * mask

    def forward(self, T, ug, vg, lat, lon, mask):
        """ Perform the geostrophic advection simulation with Euler integration and horizontal filtering.
        Args:
            T (torch.Tensor): Initial field tensor of shape (B, H, W)
            ug (torch.Tensor): Zonal velocity tensor of shape (B, H, W)
            vg (torch.Tensor): Meridional velocity tensor of shape (B, H, W)
            lat (torch.Tensor): Latitude tensor of length H (degrees)
            lon (torch.Tensor): Longitude tensor of length W (degrees)
            mask (torch.Tensor): Mask tensor of shape (H, W)
        Returns:
            torch.Tensor: Final field after integration
        """
        B, H, W = T.shape

        # Step 1: Compute dx, dy
        dlat = lat[1] - lat[0] if H > 1 else 0.5
        dlon = lon[1] - lon[0] if W > 1 else 0.5
        dy = self.R_earth * self.deg2rad * abs(dlat)  # Positive distance
        sign_y = 1 if dlat > 0 else -1  # Sign for gradient direction

        dx_base = self.R_earth * self.deg2rad * dlon  # Assuming dlon > 0
        cos_lat = torch.cos(lat * self.deg2rad)
        dx = dx_base * cos_lat.unsqueeze(1).expand(H, W)
        dx = dx.unsqueeze(0).expand(B, H, W)

        mask = mask.unsqueeze(0).expand(B, H, W) if mask.dim() == 2 else mask

        # Step 3: Euler integration with filtering
        T_current = T.clone()

        for step in range(self.steps):
            # Compute gradients of T_current
            # dT_dy with boundary handling
            dT_dy = torch.zeros_like(T_current)
            if H > 1:
                dT_dy[:, 1:-1, :] = (T_current[:, 2:, :] - T_current[:, :-2, :]) / (2 * dy)
                dT_dy[:, 0, :] = (T_current[:, 1, :] - T_current[:, 0, :]) / dy
                dT_dy[:, -1, :] = (T_current[:, -1, :] - T_current[:, -2, :]) / dy
            dT_dy *= sign_y  # Adjust sign based on lat direction

            # dT_dx with periodic boundaries using roll
            dT_dx = (torch.roll(T_current, shifts=-1, dims=2) - torch.roll(T_current, shifts=1, dims=2)) / (2 * dx)

            # Advection term
            advection = ug * dT_dx + vg * dT_dy

            # Tendency
            F_current = -advection * mask

            T_current = T_current + self.dt * F_current

            # Apply horizontal filter
            T_current = self.apply_horizontal_filter(T_current, mask)

        return T_current

# Initialize models
model_gm = SimVP(**model_config['simvp_gm_params']).to(device)
model_S = SimVP(**model_config['simvp_s_params']).to(device)
ocean_e2e = Ocean_e2e(**model_config['ocean_e2e_params']).to(device)

# Process according to the parallelization method
if parallel_method == 'DistributedDataParallel':
    model_gm = nn.parallel.DistributedDataParallel(model_gm, device_ids=[local_rank], output_device=local_rank)
    model_S = nn.parallel.DistributedDataParallel(model_S, device_ids=[local_rank], output_device=local_rank)
    # Filter and Ocean_e2e may not need DDP if they don't have trainable params, but for consistency
elif parallel_method == 'DataParallel':
    model_gm = nn.DataParallel(model_gm)
    model_S = nn.DataParallel(model_S)
else:
    raise ValueError(f"Unknown parallel method: {parallel_method}")

# ============== Loss Function and Optimizer ==============
criterion = nn.MSELoss()
optimizer = optim.Adam(list(model_gm.parameters()) + list(model_S.parameters()), lr=training_config['init_lr'])
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=training_config['lr_step_size'],
    gamma=training_config['lr_gamma']
)

# Boundary mask and lat/lon
boundary_mask = torch.tensor(np.load(data_config['boundary_path'])[:]).to(device)
boundary_mask[170:190] = 0  # As in original

lat = (90.0 - 0.25 - torch.arange(model_config['qgm_params']['nx'], device=device) * 0.5).to(device)
lon = (torch.arange(model_config['qgm_params']['ny'], device=device) * 0.5).to(device)

# ============== Training, validation, and testing functions ==============
def physics_forward(model_gm, model_S, ocean_e2e, input_era, input_uo, input_vo, input_ssta, batch_size, num_steps, lat, lon, boundary_mask):
    """执行物理驱动的多步前向模拟"""
    pred_ssta = torch.zeros([batch_size, num_steps, model_config['qgm_params']['nx'], model_config['qgm_params']['ny']]).to(device)
    # initial condition batch * H * W
    current_ssta = input_ssta[:, 0]
    for idx in range(num_steps):
        ssta = current_ssta
        uo = input_uo[:, idx]
        vo = input_vo[:, idx]
        # heat flux
        era = torch.cat([input_era[:, idx], input_era[:, idx + 1]], 1)
        # RK2 Heun's method
        vel_gm = model_gm(torch.stack([ssta, uo, vo], 1))  # B*3*H*W --> B*2*H*W
        u_gm = vel_gm[:, 0]
        v_gm = vel_gm[:, 1]
        scale_factor_gm = 0.1  # This scaling factor is used to avoid instability during training
        u_total = uo + scale_factor_gm * u_gm
        v_total = vo + scale_factor_gm * v_gm
        source = model_S(torch.cat([era, ssta.unsqueeze(1)], 1))  # B*7*H*W --> B*1*H*W
        # Replace the integration step with Ocean_e2e
        current_ssta = ocean_e2e.forward(ssta, u_total, v_total, lat, lon, boundary_mask)
        current_ssta = (current_ssta + source.squeeze(1)).float()
        pred_ssta[:, idx] = current_ssta
    return pred_ssta

def train_epoch(model_gm, model_S, ocean_e2e, train_loader, criterion, optimizer, device, epoch, lat, lon, boundary_mask):
    model_gm.train()
    model_S.train()
    if parallel_method == 'DistributedDataParallel' and train_loader.sampler is not None:
        train_loader.sampler.set_epoch(epoch)
    train_loss = 0.0
    for input_era, input_uo, input_vo, input_ssta in tqdm(train_loader, desc="Training", disable=local_rank != 0):
        input_era = input_era.to(device, non_blocking=True).float()
        input_uo = input_uo.to(device, non_blocking=True).float()
        input_vo = input_vo.to(device, non_blocking=True).float()
        input_ssta = input_ssta.to(device, non_blocking=True).float()
        optimizer.zero_grad()
        pred_ssta = physics_forward(model_gm, model_S, ocean_e2e, input_era, input_uo, input_vo, input_ssta, training_config['batch_size'], training_config['num_steps_train'], lat, lon, boundary_mask)
        loss = criterion(pred_ssta, input_ssta[:, 1:training_config['num_steps_train'] + 1])
        loss.backward()
        optimizer.step()
        loss_value = reduce_mean(loss, num_gpus).item() if parallel_method == 'DistributedDataParallel' else loss.item()
        train_loss += loss_value * input_era.size(0)
    return train_loss / len(train_loader.dataset)

def validate(model_gm, model_S, ocean_e2e, val_loader, criterion, device, lat, lon, boundary_mask):
    model_gm.eval()
    model_S.eval()
    val_loss = 0.0
    with torch.no_grad():
        for input_era, input_uo, input_vo, input_ssta in tqdm(val_loader, desc="Validation", disable=local_rank != 0):
            input_era = input_era.to(device, non_blocking=True).float()
            input_uo = input_uo.to(device, non_blocking=True).float()
            input_vo = input_vo.to(device, non_blocking=True).float()
            input_ssta = input_ssta.to(device, non_blocking=True).float()
            pred_ssta = physics_forward(model_gm, model_S, ocean_e2e, input_era, input_uo, input_vo, input_ssta, training_config['batch_size'], training_config['num_steps_train'], lat, lon, boundary_mask)
            loss = criterion(pred_ssta, input_ssta[:, 1:training_config['num_steps_train'] + 1])
            loss_value = reduce_mean(loss, num_gpus).item() if parallel_method == 'DistributedDataParallel' else loss.item()
            val_loss += loss_value * input_era.size(0)
    return val_loss / len(val_loader.dataset)

def test(model_gm, model_S, ocean_e2e, val_loader, criterion, device, lat, lon, boundary_mask):
    model_gm.eval()
    model_S.eval()
    test_loss = 0.0
    all_inputs = []
    all_outputs = []
    with torch.no_grad():
        for input_era, input_uo, input_vo, input_ssta in tqdm(val_loader, desc="Testing", disable=local_rank != 0):
            input_era = input_era.to(device, non_blocking=True).float()
            input_uo = input_uo.to(device, non_blocking=True).float()
            input_vo = input_vo.to(device, non_blocking=True).float()
            input_ssta = input_ssta.to(device, non_blocking=True).float()
            pred_ssta = physics_forward(model_gm, model_S, ocean_e2e, input_era, input_uo, input_vo, input_ssta, training_config['batch_size'], training_config['num_steps_train'], lat, lon, boundary_mask)
            if local_rank == 0:
                all_inputs.append(input_ssta.cpu().numpy())
                all_outputs.append(pred_ssta.cpu().numpy())
            loss = criterion(pred_ssta, input_ssta[:, 1:training_config['num_steps_train'] + 1])
            loss_value = reduce_mean(loss, num_gpus).item() if parallel_method == 'DistributedDataParallel' else loss.item()
            test_loss += loss_value * input_era.size(0)
    if local_rank == 0:
        all_inputs = np.concatenate(all_inputs, axis=0)
        all_outputs = np.concatenate(all_outputs, axis=0)
        np.save(f'{result_dir}/{backbone}_inputs.npy', all_inputs)
        np.save(f'{result_dir}/{backbone}_outputs.npy', all_outputs)
    return test_loss / len(val_loader.dataset)

# ============== Load Checkpoint if exists ==============
best_model_gm_path = f'{checkpoint_dir}/{backbone}_best_model_gm.pth'
best_model_S_path = f'{checkpoint_dir}/{backbone}_best_model_S.pth'

pre_model_gm_path = f'{ckpt_path_pre}_best_model_gm.pth' if ckpt_path_pre else None
pre_model_S_path = f'{ckpt_path_pre}_best_model_S.pth' if ckpt_path_pre else None

load_path_gm = None
load_path_S = None

if local_rank == 0:
    if os.path.exists(best_model_gm_path):
        load_path_gm = best_model_gm_path
        logging.info(f'Loading model_gm from {load_path_gm}.')
    elif pre_model_gm_path and os.path.exists(pre_model_gm_path):
        load_path_gm = pre_model_gm_path
        logging.info(f'Loading model_gm from pre checkpoint {load_path_gm}.')
    else:
        logging.info('No checkpoint found, training from scratch.')

    if load_path_gm:
        try:
            checkpoint = torch.load(load_path_gm, map_location=device)
            if all(k.startswith('module.') for k in checkpoint.keys()):
                checkpoint = {k[7:]: v for k, v in checkpoint.items()}
            model_gm.load_state_dict(checkpoint)
        except Exception as e:
            logging.error(f'Error loading model_gm checkpoint: {e}')

    if os.path.exists(best_model_S_path):
        load_path_S = best_model_S_path
        logging.info(f'Loading model_S from {load_path_S}.')
    elif pre_model_S_path and os.path.exists(pre_model_S_path):
        load_path_S = pre_model_S_path
        logging.info(f'Loading model_S from pre checkpoint {load_path_S}.')

    if load_path_S:
        try:
            checkpoint = torch.load(load_path_S, map_location=device)
            if all(k.startswith('module.') for k in checkpoint.keys()):
                checkpoint = {k[7:]: v for k, v in checkpoint.items()}
            model_S.load_state_dict(checkpoint)
        except Exception as e:
            logging.error(f'Error loading model_S checkpoint: {e}')

# ============== Main training Loop ==============
best_val_loss = float('inf')
num_epochs = training_config['num_epochs']
# for epoch in range(num_epochs):
#     if local_rank == 0:
#         logging.info(f'Epoch {epoch + 1}/{num_epochs}')
#     train_loss = train_epoch(model_gm, model_S, ocean_e2e, train_loader, criterion, optimizer, device, epoch, lat, lon, boundary_mask)
#     val_loss = validate(model_gm, model_S, ocean_e2e, val_loader, criterion, device, lat, lon, boundary_mask)
#     scheduler.step()
#     if local_rank == 0:
#         current_lr = optimizer.param_groups[0]['lr']
#         logging.info(f'Current Learning Rate: {current_lr:.10f}')
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(model_gm.state_dict(), best_model_gm_path)
#             torch.save(model_S.state_dict(), best_model_S_path)
#         logging.info(f'Train Loss: {train_loss:.7f}, Val Loss: {val_loss:.7f}')

# Test after training or directly if training is commented

if local_rank == 0:
    """加载预训练模型权重"""
    if not os.path.exists(best_model_gm_path):
        logging.warning(f"No checkpoint found at {best_model_gm_path}, testing with current models")
    else:
        state_dict_gm = torch.load(best_model_gm_path, map_location=device)       
        # 处理多GPU参数名前缀
        if all(k.startswith('module.') for k in state_dict_gm.keys()):
            state_dict_gm = {k[7:]: v for k, v in state_dict_gm.items()}
        
        model_gm.module.load_state_dict(state_dict_gm)
        logging.info(f"Successfully loaded checkpoint from {best_model_gm_path}")

        state_dict_S = torch.load(best_model_S_path, map_location=device)       
        # 处理多GPU参数名前缀
        if all(k.startswith('module.') for k in state_dict_S.keys()):
            state_dict_S = {k[7:]: v for k, v in state_dict_S.items()}
        
        model_S.module.load_state_dict(state_dict_S)
        logging.info(f"Successfully loaded checkpoint from {best_model_S_path}")

    test_loss = test(model_gm, model_S, ocean_e2e, val_loader, criterion, device, lat, lon, boundary_mask)
    logging.info(f"Testing completed. Test Loss: {test_loss:.7f}")

if parallel_method == 'DistributedDataParallel':
    dist.destroy_process_group()
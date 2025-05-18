# train.py
import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from config_numerical import config
from dataloader_05res import train_Dataset, test_Dataset
from baseline.UNet import UNet
from baseline.SimVP import SimVP
from baseline.DiT import Dit_plus
from baseline.AsNet import Assimilation
from utils import SmoothFilter
from baseline.Multi_conv import *
#from baseline.ERANet import ERANet
#from qgm_functional import init_qgm_state, compute_q_over_f0_from_p, step_qgm

class QGMTrainer:
    def __init__(self, device):
        # 系统配置
        self.device = device
        self.num_gpus = torch.cuda.device_count()
        self.rank = dist.get_rank()
        
        # 物理参数初始化
        #self.state = init_qgm_state(config.QGM_PARAMS)
        self.param = config.QGM_PARAMS
        
        # 模型组件
        self.model_gm, self.model_S, self.Filter = self._init_model()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(list(self.model_gm.parameters()) + list(self.model_S.parameters()),
                                          lr=config.INIT_LR)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=config.LR_STEP_SIZE,
            gamma=config.LR_GAMMA
        )
        
        # 数据加载
        self.train_loader, self.val_loader = self._prepare_dataloaders()

        # 边界mask
        self.boundary_mask = torch.tensor(np.load(config.BOUNDARY_PATH)[0:720:2,0:1440:2]).to(self.device)
        self.boundary_mask[170:190] = 0

    def _init_model(self):
        """初始化并包装DDP模型"""
        #model_gm = UNet(**config.MODEL_GM).to(self.device)
        #model_S = UNet(**config.MODEL_S).to(self.device)
        #model = ERANet().to(self.device)
        # model_gm = Dit_plus(shape_in=(1, 1, 3, 360, 720), shape_out=(1, 1, 2, 360, 720), hid_S=32, hid_T=64, N_S=4, N_T=8, time_step=1000).to(self.device)
        # model_S = Dit_plus(shape_in=(1, 1, 7, 360, 720), shape_out=(1, 1, 1, 360, 720), hid_S=32, hid_T=64, N_S=4, N_T=8, time_step=1000).to(self.device)
        # model_gm = Assimilation(in_c=3, out_c=2).to(self.device)
        # model_S = Assimilation(in_c=7, out_c=1).to(self.device)
        # model_gm = MultiConv(in_shape=(1, 2, 360, 720)).to(self.device)
        # model_S = MultiConv(in_shape=(1, 7, 360, 720), out_shape=(1, 1, 360, 720)).to(self.device)
        model_gm = SimVP(shape_in=[1, 3, 360, 720], shape_out=[1, 2, 360, 720]).to(self.device)
        model_S = SimVP(shape_in=[1, 7, 360, 720], shape_out=[1, 1, 360, 720]).to(self.device)
        Filter = SmoothFilter(channels=1).to(self.device)
        return torch.nn.parallel.DistributedDataParallel(model_gm, device_ids=[self.device.index], broadcast_buffers=False, find_unused_parameters=False), torch.nn.parallel.DistributedDataParallel(model_S, device_ids=[self.device.index], broadcast_buffers=False, find_unused_parameters=False), Filter

    def _prepare_dataloaders(self):
        """创建分布式数据加载器"""
        train_set = train_Dataset(config.DATA)
        val_set = test_Dataset(config.DATA)
        
        return (
            DataLoader(
                train_set,
                batch_size=config.BATCH_SIZE,
                sampler=DistributedSampler(train_set),
                num_workers=config.NUM_WORKERS,
                pin_memory=True
            ),
            DataLoader(
                val_set,
                batch_size=config.BATCH_SIZE,
                sampler=DistributedSampler(val_set,shuffle=False),
                num_workers=config.NUM_WORKERS,
                pin_memory=True,
                shuffle=False
            )
        )

    def _sync_tensor(self, tensor):
        """跨GPU张量同步"""
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        return rt / self.num_gpus
        
    def _cal_x_grad(self, input_field, EARTH_RADIUS=6371e3 / (24*3600)):
        """
        计算经线方向（东向）的梯度
        Args:
            input_field (Tensor): 输入场 [B, H, W] 或 [H, W]，纬度从北到南排列
        Returns:
            Tensor: 东向梯度场，单位：输入单位/米
        """
        # 生成纬度数组（单位：度）
        lat_deg = 90.0 - 0.25 - torch.arange(360, device=input_field.device) * 0.5
        lat_rad = torch.deg2rad(lat_deg)
    
        # 计算经向格距（东向）
        dx = EARTH_RADIUS * torch.cos(lat_rad.unsqueeze(1)) * torch.deg2rad(torch.tensor(0.5, device=input_field.device))
        #print(dx)
    
        # 周期边界处理
        left = torch.roll(input_field, shifts=1, dims=-1)  # 左邻居（经度维度）
        right = torch.roll(input_field, shifts=-1, dims=-1)  # 右邻居
    
        # 中心差分计算
        return (right - left) / (2 * dx)

    def _cal_y_grad(self, input_field, EARTH_RADIUS=6371e3 / (24*3600)):
        """
        计算纬线方向（北向）的梯度
        Args:
            input_field (Tensor): 输入场 [B, H, W] 或 [H, W]，纬度从北到南排列
        Returns:
            Tensor: 北向梯度场，单位：输入单位/米
        """
        # 固定纬向格距
        dy = EARTH_RADIUS * torch.deg2rad(torch.tensor(0.5, device=input_field.device))
        #print(dy)
    
        # 初始化梯度场
        grad = torch.zeros_like(input_field)
    
        # 中间区域中心差分
        if input_field.dim() == 2:
            grad[1:-1, :] = (-input_field[:-2, :] + input_field[2:, :]) / (2 * dy)
            # 边界处理
            grad[0, :] = (-input_field[0, :] + input_field[1, :]) / dy  # 北极
            grad[-1, :] = (-input_field[-2, :] + input_field[-1, :]) / dy  # 南极
        else:
            grad[:, 1:-1, :] = (-input_field[:, :-2, :] + input_field[:, 2:, :]) / (2 * dy)
            grad[:, 0, :] = (-input_field[:, 0, :] + input_field[:, 1, :]) / dy
            grad[:, -1, :] = (-input_field[:, -2, :] + input_field[:, -1, :]) / dy
    
        return grad

    def _physics_forward(self, input_era, input_uo, input_vo, input_ssta, batch_size, num_steps):
        """执行物理驱动的多步前向模拟"""
        pred_ssta = torch.zeros([batch_size, num_steps, self.param['nx'], self.param['ny']]).to(self.device)
        #initial condition batch * H * W
        current_ssta = input_ssta[:,0]

        for idx in range(num_steps):
            #print('1', current_ssta.shape)
            #ssta_x = self._cal_x_grad(current_ssta)
            #ssta_y = self._cal_y_grad(current_ssta)
            ssta = current_ssta
            uo = input_uo[:,idx]
            vo = input_vo[:,idx]
            # heat flux
            era = torch.cat([input_era[:,idx], input_era[:,idx+1]], 1)
            
            # RK2 Heun's method
            vel_gm = self.model_gm(torch.stack([ssta, uo, vo], 1)) # B*3*H*W --> B*2*H*W
            #vel_gm = self.model_gm(torch.concat([torch.stack([ssta, uo, vo], 1), era], 1)) # B*9*H*W --> B*2*H*W
            # # kappa = self.model_K(torch.stack([ssta, uo, vo], 1))
            # # k11 = kappa[:,0]
            # # k12 = kappa[:,1]
            # # k22 = kappa[:,2]
            
            u_gm = vel_gm[:,0]
            v_gm = vel_gm[:,1]
            scale_factor_gm = 0.1
            # scale_factor_k = 0.01
            adv = - self._cal_x_grad(((uo+scale_factor_gm*u_gm) * ssta)) - self._cal_y_grad(((vo+scale_factor_gm*v_gm) * ssta))
            source = self.model_S(torch.cat([era, ssta.unsqueeze(1)],1)) # B*7*H*W --> B*1*H*W
            # dis = self._cal_x_grad(k11 * scale_factor_k * self._cal_x_grad(ssta)) + self._cal_y_grad(k22 * scale_factor_k * self._cal_y_grad(ssta)) + self._cal_x_grad(k12 * scale_factor_k * self._cal_y_grad(ssta)) + self._cal_y_grad(k12 * scale_factor_k * self._cal_x_grad(ssta))
            # delta_ssta_1 = self.Filter(adv.unsqueeze(1)).squeeze(0)*self.boundary_mask
            delta_ssta_1 = source.squeeze(1)
            # delta_ssta_1 = self.Filter(adv.unsqueeze(1)).squeeze(0)*self.boundary_mask + source.squeeze(1)
            ssta = ssta + delta_ssta_1
            current_ssta = ssta

            # vel_gm = self.model_gm(torch.stack([ssta, uo, vo], 1)) # B*3*H*W --> B*2*H*W
            # u_gm = vel_gm[:,0]
            # v_gm = vel_gm[:,1]
            # adv = - self._cal_x_grad(((uo+scale_factor*u_gm) * ssta)) - self._cal_y_grad(((vo+scale_factor*v_gm) * ssta))
            # source = self.model_S(torch.cat([era, ssta.unsqueeze(1)],1)) # B*7*H*W --> B*1*H*W
            # delta_ssta_2 = (adv)*self.boundary_mask + source.squeeze(1)
            # current_ssta = current_ssta + 0.5 * (delta_ssta_1 + delta_ssta_2)
            
            pred_ssta[:,idx] = current_ssta
            
        return pred_ssta
        
    def _train_epoch(self):
        """单epoch训练"""
        self.model_gm.train()
        self.model_S.train()
        total_loss = 0.0
        
        for input_era, input_uo, input_vo, input_ssta in tqdm(self.train_loader, desc="Training", disable=self.rank!=0):
            input_era = input_era.to(self.device, non_blocking=True).float()
            input_uo = input_uo.to(self.device, non_blocking=True).float()
            input_vo = input_vo.to(self.device, non_blocking=True).float()
            #input_mld = input_mld.to(self.device, non_blocking=True).float()
            input_ssta = input_ssta.to(self.device, non_blocking=True).float()
            
            self.optimizer.zero_grad()
            # 多步物理模拟
            pred_ssta = self._physics_forward(input_era, input_uo, input_vo, input_ssta, config.BATCH_SIZE, config.NUM_STEPS_TRAIN)
            #print(pred_ssta.shape, input_ssta[:,1:].shape)
            loss = self.criterion(pred_ssta, input_ssta[:,1:config.NUM_STEPS_TRAIN+1])
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 损失同步
            sync_loss = self._sync_tensor(loss)
            total_loss += sync_loss.item() * input_era.size(0)
            
        return total_loss / len(self.train_loader.dataset)

    def _validate(self):
        """验证流程"""
        self.model_gm.eval()
        self.model_S.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for input_era, input_uo, input_vo, input_ssta in tqdm(self.val_loader, desc="Validating", disable=self.rank!=0):
                input_era = input_era.to(self.device, non_blocking=True).float()
                input_uo = input_uo.to(self.device, non_blocking=True).float()
                input_vo = input_vo.to(self.device, non_blocking=True).float()
                #input_mld = input_mld.to(self.device, non_blocking=True).float()
                input_ssta = input_ssta.to(self.device, non_blocking=True).float()
                
                pred_ssta = self._physics_forward(input_era, input_uo, input_vo, input_ssta, config.BATCH_SIZE, config.NUM_STEPS_TRAIN)
                #print(pred_ssta.shape, input_ssta[:,1:].shape)
                loss = self.criterion(pred_ssta, input_ssta[:,1:config.NUM_STEPS_TRAIN+1])
                
                sync_loss = self._sync_tensor(loss)
                total_loss += sync_loss.item() * input_era.size(0)
                
        return total_loss / len(self.val_loader.dataset)

    def _load_checkpoint(self):
        """加载预训练模型权重"""
        ckpt_path_gm = f"{config.CKPT_PATH_PRE}_best_model_gm.pth"
        ckpt_path_S = f"{config.CKPT_PATH_PRE}_best_model_S.pth"
        # ckpt_path_K = f"{config.CKPT_PATH_PRE}_best_model_K.pth"
        if not os.path.exists(ckpt_path_gm):
            if self.rank == 0:
                logging.warning(f"No checkpoint found at {ckpt_path_gm}, training from scratch")
            return

        if self.rank == 0:
            state_dict_gm = torch.load(ckpt_path_gm, map_location=self.device)       
            # 处理多GPU参数名前缀
            if all(k.startswith('module.') for k in state_dict_gm.keys()):
                state_dict_gm = {k[7:]: v for k, v in state_dict_gm.items()}
            
            self.model_gm.module.load_state_dict(state_dict_gm)
            logging.info(f"Successfully loaded checkpoint from {ckpt_path_gm}")

            state_dict_S = torch.load(ckpt_path_S, map_location=self.device)       
            # 处理多GPU参数名前缀
            if all(k.startswith('module.') for k in state_dict_S.keys()):
                state_dict_S = {k[7:]: v for k, v in state_dict_S.items()}
            
            self.model_S.module.load_state_dict(state_dict_S)
            logging.info(f"Successfully loaded checkpoint from {ckpt_path_S}")

            # state_dict_K = torch.load(ckpt_path_K, map_location=self.device)       
            # # 处理多GPU参数名前缀
            # if all(k.startswith('module.') for k in state_dict_K.keys()):
            #     state_dict_K = {k[7:]: v for k, v in state_dict_K.items()}
            
            # self.model_K.module.load_state_dict(state_dict_K)
            # logging.info(f"Successfully loaded checkpoint from {ckpt_path_K}")
        
    def _test(self):
        """测试流程"""
        self.model_S.eval()
        self.model_gm.eval()
        test_loss = 0.0
        all_inputs = []
        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for input_era, input_uo, input_vo, input_ssta in tqdm(self.val_loader, desc="Validating", disable=self.rank!=0):
                input_era = input_era.to(self.device, non_blocking=True).float()
                input_uo = input_uo.to(self.device, non_blocking=True).float()
                input_vo = input_vo.to(self.device, non_blocking=True).float()
                #input_mld = input_mld.to(self.device, non_blocking=True).float()
                input_ssta = input_ssta.to(self.device, non_blocking=True).float()
                
                pred_ssta = self._physics_forward(input_era, input_uo, input_vo, input_ssta, config.BATCH_SIZE, config.NUM_STEPS_TRAIN)
            
                # Convert tensors to numpy arrays and append to lists
                all_inputs.append(input_ssta.cpu().numpy())
                all_outputs.append(pred_ssta.cpu().numpy())
                
        all_inputs = np.concatenate(all_inputs, axis=0)
        all_outputs = np.concatenate(all_outputs, axis=0)

        np.save(f'./results/{config.BACKBONE}_inputs.npy', all_inputs)
        np.save(f'./results/{config.BACKBONE}_outputs.npy', all_outputs)
        

    def run(self):
        """主训练循环"""
        best_loss = float('inf')

        self._load_checkpoint()
        self._test()
        
        # for epoch in range(config.EPOCHS):
        #     # 设置分布式采样器
        #     self.train_loader.sampler.set_epoch(epoch)
            
        #     # 训练阶段
        #     train_loss = self._train_epoch()
        #     self.scheduler.step()
            
        #     # 验证阶段
        #     val_loss = self._validate()
            
        #     # 主进程保存结果
        #     if self.rank == 0:
        #         self._save_checkpoint(val_loss < best_loss)
        #         best_loss = min(val_loss, best_loss)
        #         self._log_progress(epoch, train_loss, val_loss)

    def _save_checkpoint(self, is_best):
        """保存模型检查点"""
        if is_best:
            torch.save(self.model_gm.module.state_dict(), f"{config.CKPT_PATH}_best_model_gm.pth")
            torch.save(self.model_S.module.state_dict(), f"{config.CKPT_PATH}_best_model_S.pth")

    def _log_progress(self, epoch, train_loss, val_loss):
        """记录训练进度"""
        lr = self.optimizer.param_groups[0]['lr']
        log_msg = (
            f"Epoch {epoch+1:03d} | "
            f"LR: {lr:.2e} | "
            f"Train: {train_loss*self.num_gpus:.7f} | "
            f"Val: {val_loss*self.num_gpus:.7f}"
        )
        logging.info(log_msg)
        print(log_msg)  # 控制台输出

def main():
    """主函数"""
    # 分布式环境初始化
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    # 设置随机种子
    torch.manual_seed(config.SEED + local_rank)
    np.random.seed(config.SEED + local_rank)

    if local_rank == 0:
        logging.info(f"Training configuration:\n{vars(config)}")
        os.makedirs(config.CKPT_PATH, exist_ok=True)
        
    trainer = QGMTrainer(device)
    trainer.run()
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
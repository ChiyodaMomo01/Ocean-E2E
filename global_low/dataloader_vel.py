# dataloader_vel.py
import numpy as np
import netCDF4 as nc
import torch
import torch.utils.data as data

class train_Dataset(data.Dataset):
    def __init__(self, data_path):
        super(train_Dataset, self).__init__()
        self.data_path = data_path
        self.atmosphere_lead_time = 10
        self.ds_factor = 2
        self.lat_start = 0
        self.lat_end = 720
        self.lon_start = 0
        self.lon_end = 1440
        self.years = range(1993, 2019)
        self.dates = range(10, 350, 1)
        self.indices = [(m, n) for m in self.years for n in self.dates]
       
    def __getitem__(self, index):
        years, dates = self.indices[index]
        train_data_altimetry = nc.Dataset(f'{self.data_path}/low/SSH_low_processed_{years}.nc')
        input_uo = train_data_altimetry.variables['ugos'][dates-self.atmosphere_lead_time:dates,
                                   self.lat_start:self.lat_end:self.ds_factor,
                                   self.lon_start:self.lon_end:self.ds_factor]
        input_vo = train_data_altimetry.variables['vgos'][dates-self.atmosphere_lead_time:dates,
                                   self.lat_start:self.lat_end:self.ds_factor,
                                   self.lon_start:self.lon_end:self.ds_factor]
        target_uo = train_data_altimetry.variables['ugos'][dates:dates+self.atmosphere_lead_time,
                                   self.lat_start:self.lat_end:self.ds_factor,
                                   self.lon_start:self.lon_end:self.ds_factor]
        target_vo = train_data_altimetry.variables['vgos'][dates:dates+self.atmosphere_lead_time,
                                   self.lat_start:self.lat_end:self.ds_factor,
                                   self.lon_start:self.lon_end:self.ds_factor]
        input_uo = torch.tensor(input_uo.filled())
        input_vo = torch.tensor(input_vo.filled())
        target_uo = torch.tensor(target_uo.filled())
        target_vo = torch.tensor(target_vo.filled())
       
        input_uo = torch.nan_to_num(input_uo, nan=0.0)
        input_vo = torch.nan_to_num(input_vo, nan=0.0)
        target_uo = torch.nan_to_num(target_uo, nan=0.0)
        target_vo = torch.nan_to_num(target_vo, nan=0.0)
        input = torch.stack([input_uo, input_vo], 1)
        target = torch.stack([target_uo, target_vo], 1)
       
        return input, target
    def __len__(self):
        return len(self.indices)
   
class test_Dataset(data.Dataset):
    def __init__(self, data_path):
        super(test_Dataset, self).__init__()
        self.data_path = data_path
        self.atmosphere_lead_time = 10
        self.ds_factor = 2
        self.lat_start = 0
        self.lat_end = 720
        self.lon_start = 0
        self.lon_end = 1440
        self.years = range(2019, 2020)
        self.dates = range(12, 300, 3)
        self.indices = [(m, n) for m in self.years for n in self.dates]
       
    def __getitem__(self, index):
        years, dates = self.indices[index]
        print(dates)
        train_data_altimetry = nc.Dataset(f'{self.data_path}/low/SSH_low_processed_{years}.nc')
        input_uo = train_data_altimetry.variables['ugos'][dates-self.atmosphere_lead_time:dates,
                                   self.lat_start:self.lat_end:self.ds_factor,
                                   self.lon_start:self.lon_end:self.ds_factor]
        input_vo = train_data_altimetry.variables['vgos'][dates-self.atmosphere_lead_time:dates,
                                   self.lat_start:self.lat_end:self.ds_factor,
                                   self.lon_start:self.lon_end:self.ds_factor]
        target_uo = train_data_altimetry.variables['ugos'][dates:dates+self.atmosphere_lead_time,
                                   self.lat_start:self.lat_end:self.ds_factor,
                                   self.lon_start:self.lon_end:self.ds_factor]
        target_vo = train_data_altimetry.variables['vgos'][dates:dates+self.atmosphere_lead_time,
                                   self.lat_start:self.lat_end:self.ds_factor,
                                   self.lon_start:self.lon_end:self.ds_factor]
        input_uo = torch.tensor(input_uo.filled())
        input_vo = torch.tensor(input_vo.filled())
        target_uo = torch.tensor(target_uo.filled())
        target_vo = torch.tensor(target_vo.filled())
       
        input_uo = torch.nan_to_num(input_uo, nan=0.0)
        input_vo = torch.nan_to_num(input_vo, nan=0.0)
        target_uo = torch.nan_to_num(target_uo, nan=0.0)
        target_vo = torch.nan_to_num(target_vo, nan=0.0)
        input = torch.stack([input_uo, input_vo], 1)
        target = torch.stack([target_uo, target_vo], 1)
       
        return input, target
    def __len__(self):
        return len(self.indices)
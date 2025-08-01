import netCDF4 as nc 
import numpy as np
import torch
import torch.utils.data as data_utils

class train_Dataset(data_utils.Dataset):
    def __init__(self, data_path):
        super(train_Dataset, self).__init__()
        self.years = range(1993, 2019)
        self.dates = range(10, 350, 3)
        self.data_path = data_path
        self.indices = [(m, n) for m in self.years for n in self.dates]
        
    def __getitem__(self, index):
        #print(index)
        years, dates = self.indices[index]
        train_data_vel = nc.Dataset(f'{self.data_path}/geo_{years}.nc')
        input_uo = train_data_vel.variables['mhws_variables'][dates:dates+5,0].filled(np.nan)
        input_vo = train_data_vel.variables['mhws_variables'][dates:dates+5,1].filled(np.nan)
        
        train_data_mhw = nc.Dataset(f'{self.data_path}/mhw_{years}.nc')
        input_ssta = train_data_mhw.variables['mhws_variables'][dates:dates+5,0].filled(np.nan)
        input_era = train_data_mhw.variables['mhws_variables'][dates:dates+5,[2,3,4]].filled(np.nan)
        
        input_uo = torch.tensor(input_uo)
        input_vo = torch.tensor(input_vo)
        input_ssta = torch.tensor(input_ssta)
        input_era = torch.tensor(input_era)
        input_uo = torch.nan_to_num(input_uo, nan=0)
        input_vo = torch.nan_to_num(input_vo, nan=0)
        input_ssta = torch.nan_to_num(input_ssta, nan=0)
        input_era = torch.nan_to_num(input_era, nan=0)
        return input_era, input_uo, input_vo, input_ssta

    def __len__(self):
        return len(self.indices)
    
# class test_Dataset(data_utils.Dataset):
#     def __init__(self, data_path):
#         super(test_Dataset, self).__init__()
#         self.years = range(2018, 2019)
#         self.dates = range(10, 350, 3)
#         self.data_path = data_path
#         self.indices = [(m, n) for m in self.years for n in self.dates]
        
#     def __getitem__(self, index):
#         #print(index)
#         years, dates = self.indices[index]
#         train_data_vel = nc.Dataset(f'{self.data_path}/geo_{years}.nc')
#         input_uo = train_data_vel.variables['mhws_variables'][dates:dates+5,0].filled(np.nan)
#         input_vo = train_data_vel.variables['mhws_variables'][dates:dates+5,1].filled(np.nan)
        
#         train_data_mhw = nc.Dataset(f'{self.data_path}/mhw_{years}.nc')
#         input_ssta = train_data_mhw.variables['mhws_variables'][dates:dates+5,0].filled(np.nan)
#         input_era = train_data_mhw.variables['mhws_variables'][dates:dates+5,[2,3,4]].filled(np.nan)
        
#         input_uo = torch.tensor(input_uo)
#         input_vo = torch.tensor(input_vo)
#         input_ssta = torch.tensor(input_ssta)
#         input_era = torch.tensor(input_era)
#         input_uo = torch.nan_to_num(input_uo, nan=0)
#         input_vo = torch.nan_to_num(input_vo, nan=0)
#         input_ssta = torch.nan_to_num(input_ssta, nan=0)
#         input_era = torch.nan_to_num(input_era, nan=0)
#         return input_era, input_uo, input_vo, input_ssta

#     def __len__(self):
#         return len(self.indices)
    
    
class test_Dataset(data_utils.Dataset):
    def __init__(self, data_path):
        super(test_Dataset, self).__init__()
        self.years = range(2020, 2021)
        self.dates = range(10, 240, 1)
        self.data_path = data_path
        self.indices = [(m, n) for m in self.years for n in self.dates]
        
    def __getitem__(self, index):
        print(index)
        years, dates = self.indices[index]
        # train_data_vel = nc.Dataset(f'{self.data_path}/geo_{years}.nc')
        # input_uo = train_data_vel.variables['mhws_variables'][dates:dates+31,0].filled(np.nan)
        # input_vo = train_data_vel.variables['mhws_variables'][dates:dates+31,1].filled(np.nan)
        
        train_data_vel = nc.Dataset(f'{self.data_path}/MultiConv_data.nc')
        input_uo = train_data_vel.variables['outputs'][dates-10,:,0].filled(np.nan)
        input_vo = train_data_vel.variables['outputs'][dates-10,:,1].filled(np.nan)
        
        train_data_mhw = nc.Dataset(f'{self.data_path}/mhw_{years}.nc')
        input_ssta = train_data_mhw.variables['mhws_variables'][dates:dates+31,0].filled(np.nan)
        input_era = train_data_mhw.variables['mhws_variables'][dates:dates+31,[2,3,4]].filled(np.nan)
        
        input_uo = torch.tensor(input_uo)
        input_vo = torch.tensor(input_vo)
        input_ssta = torch.tensor(input_ssta)
        input_era = torch.tensor(input_era)
        input_uo = torch.nan_to_num(input_uo, nan=0)
        input_vo = torch.nan_to_num(input_vo, nan=0)
        input_ssta = torch.nan_to_num(input_ssta, nan=0)
        input_era = torch.nan_to_num(input_era, nan=0)
        return input_era, input_uo, input_vo, input_ssta

    def __len__(self):
        return len(self.indices)
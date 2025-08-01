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
        # train_data_vel = nc.Dataset(f'{self.data_path}/geo_{years}.nc')
        # input_uo = train_data_vel.variables['mhws_variables'][dates:dates+5,0].filled(np.nan)
        # input_vo = train_data_vel.variables['mhws_variables'][dates:dates+5,1].filled(np.nan)
        
        train_data = nc.Dataset(f'{self.data_path}/mhw_{years}.nc')
        input_now = train_data.variables['mhws_variables'][dates,[0,2,3,4]].filled(np.nan)
        input_future = train_data.variables['mhws_variables'][dates+1,[2,3,4]].filled(np.nan)
        input = np.concatenate([input_now, input_future], 0)
        target = train_data.variables['mhws_variables'][dates+1,[0]].filled(np.nan)
        
        input = torch.tensor(input)
        target = torch.tensor(target)
        input = torch.nan_to_num(input, nan=0)
        target = torch.nan_to_num(target, nan=0)
        return input.unsqueeze(0), target.unsqueeze(0)

    def __len__(self):
        return len(self.indices)
    
# class test_Dataset(data_utils.Dataset):
#     def __init__(self, data_path):
#         super(test_Dataset, self).__init__()
#         self.years = range(2019, 2020)
#         self.dates = range(10, 350, 3)
#         self.data_path = data_path
#         self.indices = [(m, n) for m in self.years for n in self.dates]
        
#     def __getitem__(self, index):
#         #print(index)
#         years, dates = self.indices[index]
#         # train_data_vel = nc.Dataset(f'{self.data_path}/geo_{years}.nc')
#         # input_uo = train_data_vel.variables['mhws_variables'][dates:dates+5,0].filled(np.nan)
#         # input_vo = train_data_vel.variables['mhws_variables'][dates:dates+5,1].filled(np.nan)
        
#         train_data = nc.Dataset(f'{self.data_path}/mhw_{years}.nc')
#         input_now = train_data.variables['mhws_variables'][dates,[0,2,3,4]].filled(np.nan)
#         input_future = train_data.variables['mhws_variables'][dates+1,[2,3,4]].filled(np.nan)
#         input = np.concatenate([input_now, input_future], 0)
#         target = train_data.variables['mhws_variables'][dates+1,[0]].filled(np.nan)
        
#         input = torch.tensor(input)
#         target = torch.tensor(target)
#         input = torch.nan_to_num(input, nan=0)
#         target = torch.nan_to_num(target, nan=0)
#         return input.unsqueeze(0), target.unsqueeze(0)

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
        #print(index)
        years, dates = self.indices[index]
        train_data = nc.Dataset(f'{self.data_path}/mhw_{years}.nc')
        input_mhw = train_data.variables['mhws_variables'][dates:dates+31,[0]].filled(np.nan)
        input_future = train_data.variables['mhws_variables'][dates:dates+31,[2,3,4]].filled(np.nan)
        
        input_mhw = torch.tensor(input_mhw)
        input_future = torch.tensor(input_future)
        input_mhw = torch.nan_to_num(input_mhw, nan=0)
        input_future = torch.nan_to_num(input_future, nan=0)
        return input_mhw, input_future

    def __len__(self):
        return len(self.indices)
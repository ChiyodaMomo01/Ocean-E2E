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
        train_data = nc.Dataset(f'{self.data_path}/geo_{years}.nc')
        input = train_data.variables['mhws_variables'][dates-1:dates,:].filled(np.nan)
        target = train_data.variables['mhws_variables'][dates:dates+1,:].filled(np.nan)
        
        input = torch.tensor(input)
        target = torch.tensor(target)
        input = torch.nan_to_num(input, nan=0)
        target = torch.nan_to_num(target, nan=0)
        return input, target

    def __len__(self):
        return len(self.indices)
    
class test_Dataset(data_utils.Dataset):
    def __init__(self, data_path):
        super(test_Dataset, self).__init__()
        self.years = range(2019, 2020)
        self.dates = range(10, 350, 3)
        self.data_path = data_path
        self.indices = [(m, n) for m in self.years for n in self.dates]
        
    def __getitem__(self, index):
        #print(index)
        years, dates = self.indices[index]
        train_data = nc.Dataset(f'{self.data_path}/geo_{years}.nc')
        input = train_data.variables['mhws_variables'][dates-1:dates,:,0:-1:2,0:-1:2].filled(np.nan)
        target = train_data.variables['mhws_variables'][dates:dates+1,:,0:-1:2,0:-1:2].filled(np.nan)
        
        input = torch.tensor(input)
        target = torch.tensor(target)
        input = torch.nan_to_num(input, nan=0)
        target = torch.nan_to_num(target, nan=0)
        return input, target

    def __len__(self):
        return len(self.indices)

# class test_Dataset(data_utils.Dataset):
#     def __init__(self, data_path):
#         super(test_Dataset, self).__init__()
#         self.years = range(2020, 2021)
#         self.dates = range(10, 240, 10)
#         self.data_path = data_path
#         self.indices = [(m, n) for m in self.years for n in self.dates]
        
#     def __getitem__(self, index):
#         print(index)
#         years, dates = self.indices[index]
#         train_data = nc.Dataset(f'{self.data_path}/geo_{years}.nc')
#         input = train_data.variables['mhws_variables'][dates-10:dates,:].filled(np.nan)
#         target = train_data.variables['mhws_variables'][dates:dates+30,:].filled(np.nan)
        
#         input = torch.tensor(input)
#         target = torch.tensor(target)
#         input = torch.nan_to_num(input, nan=0)
#         target = torch.nan_to_num(target, nan=0)
#         return input, target

#     def __len__(self):
#         return len(self.indices)
import numpy as np
import netCDF4 as nc
import torch
import torch.utils.data as data

class train_Dataset(data.Dataset):
    def __init__(self, args):
        super(train_Dataset, self).__init__()
        self.years = range(1993, 2019)
        self.dates = range(12, 357, 3)
        self.indices = [(m, n) for m in self.years for n in self.dates]
        
    def __getitem__(self, index):
        years, dates = self.indices[index]
        train_data_era = nc.Dataset(f'./data/ERA5/025res_{years}.nc')
        train_data_glorys = nc.Dataset(f'./data/highres/CMEMS_{years}_norm.nc')
        train_data_altimetry = nc.Dataset(f'./data/low/SSH_low_processed_{years}.nc')
        input_era = train_data_era.variables['mhws_variables'][dates:dates+61, 
                                   [2,3,4],0:-1:2,0:-1:2]
        
        input_uo = train_data_altimetry.variables['ugos'][dates:dates+61,0:-1:2,0:-1:2]

        input_vo = train_data_altimetry.variables['vgos'][dates:dates+61,0:-1:2,0:-1:2]
        
        input_ssta = train_data_glorys.variables['ssta_highres'][dates:dates+61,0:-1:2,0:-1:2]
        
        input_era = torch.tensor(input_era.filled())
        input_uo = torch.tensor(input_uo.filled())
        input_vo = torch.tensor(input_vo.filled())
        input_ssta = torch.tensor(input_ssta.filled())
        
        input_era = torch.nan_to_num(input_era, nan=0.0)
        input_uo = torch.nan_to_num(input_uo, nan=0.0)
        input_vo = torch.nan_to_num(input_vo, nan=0.0)
        input_ssta = torch.nan_to_num(input_ssta, nan=0.0)
        
        return input_era, input_uo, input_vo, input_ssta

    def __len__(self):
        return len(self.indices)
    
class test_Dataset(data.Dataset):
    def __init__(self, args):
        super(test_Dataset, self).__init__()
        self.args = args
        self.years = range(2020, 2021)
        self.dates = range(12, 300, 3)
        self.indices = [(m, n) for m in self.years for n in self.dates]
        
    def __getitem__(self, index):
        years, dates = self.indices[index]
        train_data_era = nc.Dataset(f'./data/ERA5/025res_{years}.nc')
        train_data_glorys = nc.Dataset(f'./data/highres/CMEMS_{years}_norm.nc')
        train_data_altimetry = nc.Dataset(f'./data/low/SSH_low_processed_{years}.nc')
        input_era = train_data_era.variables['mhws_variables'][dates:dates+61, 
                                   [2,3,4],0:-1:2,0:-1:2]
        
        input_uo = train_data_altimetry.variables['ugos'][dates:dates+61,0:-1:2,0:-1:2]

        input_vo = train_data_altimetry.variables['vgos'][dates:dates+61,0:-1:2,0:-1:2]
        
        input_ssta = train_data_glorys.variables['ssta_highres'][dates:dates+61,0:-1:2,0:-1:2]
        
        input_era = torch.tensor(input_era.filled())
        input_uo = torch.tensor(input_uo.filled())
        input_vo = torch.tensor(input_vo.filled())
        input_ssta = torch.tensor(input_ssta.filled())
        
        input_era = torch.nan_to_num(input_era, nan=0.0)
        input_uo = torch.nan_to_num(input_uo, nan=0.0)
        input_vo = torch.nan_to_num(input_vo, nan=0.0)
        input_ssta = torch.nan_to_num(input_ssta, nan=0.0)
        
        return input_era, input_uo, input_vo, input_ssta

    def __len__(self):
        return len(self.indices)
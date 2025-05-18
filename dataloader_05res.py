import numpy as np
import netCDF4 as nc
import torch
import torch.utils.data as data

args = {
    'data_path': '/apdcephfs_qy3/share_301734960/easyluwu/shuruiqi/NeuralPS_25',
    'forecast_path': '/data1/shuruiqi/Projects/MHWs/data/NC_files/Pangu',
    'operational': False,
    'ocean_lead_time': 30,
    'atmosphere_lead_time': 10,
    'shuffle': True,
    'variables_input': [0, 2, 3, 4],
    'variables_future': [2, 3, 4],
    'variables_output': [0],
    'lon_start': 0,
    'lat_start': 0,
    'lon_end': 1440,
    'lat_end': 720,
    'ds_factor': 2,
}

class train_Dataset(data.Dataset):
    def __init__(self, args):
        super(train_Dataset, self).__init__()
        self.args = args
        self.years = range(1993, 2019)
        self.dates = range(12, 357, 3)
        self.indices = [(m, n) for m in self.years for n in self.dates]
        
    def __getitem__(self, index):
        years, dates = self.indices[index]
        train_data_era = nc.Dataset(f'{self.args["data_path"]}/ERA5/025res_{years}.nc')
        #train_data_cmems = nc.Dataset(f'{self.args["data_path"]}/CMEMS/low/CMEMS_low_processed_{years}.nc')
        train_data_glorys = nc.Dataset(f'/jizhicfs/easyluwu/ocean_project/ft_local/ocean_mhws/shuruiqi_code/ft_local/highres/CMEMS_{years}_norm.nc')
        train_data_altimetry = nc.Dataset(f'{self.args["data_path"]}/CMEMS/low/SSH_low_processed_{years}.nc')
        input_era = train_data_era.variables['mhws_variables'][dates:dates+self.args['atmosphere_lead_time']+1, 
                                   [2,3,4],
                                   self.args['lat_start']:self.args['lat_end']:self.args['ds_factor'],
                                   self.args['lon_start']:self.args['lon_end']:self.args['ds_factor']]
        
        input_uo = train_data_altimetry.variables['ugos'][dates:dates+self.args['atmosphere_lead_time']+1,
                                   self.args['lat_start']:self.args['lat_end']:self.args['ds_factor'],
                                   self.args['lon_start']:self.args['lon_end']:self.args['ds_factor']]

        input_vo = train_data_altimetry.variables['vgos'][dates:dates+self.args['atmosphere_lead_time']+1,
                                   self.args['lat_start']:self.args['lat_end']:self.args['ds_factor'],
                                   self.args['lon_start']:self.args['lon_end']:self.args['ds_factor']]

        #input_mld = train_data_cmems.variables['mld'][dates:dates+self.args['atmosphere_lead_time']+1,
                                   #self.args['lat_start']:self.args['lat_end']:self.args['ds_factor'],
                                   #self.args['lon_start']:self.args['lon_end']:self.args['ds_factor']]
        input_ssta = train_data_glorys.variables['ssta_highres'][dates:dates+self.args['ocean_lead_time']+1,
                                   self.args['lat_start']:self.args['lat_end']:self.args['ds_factor'],
                                   self.args['lon_start']:self.args['lon_end']:self.args['ds_factor']]
        
        input_era = torch.tensor(input_era.filled())
        input_uo = torch.tensor(input_uo.filled())
        input_vo = torch.tensor(input_vo.filled())
        #input_mld = torch.tensor(input_mld.filled())
        input_ssta = torch.tensor(input_ssta.filled())
        
        input_era = torch.nan_to_num(input_era, nan=0.0)
        input_uo = torch.nan_to_num(input_uo, nan=0.0)
        input_vo = torch.nan_to_num(input_vo, nan=0.0)
        #input_mld = torch.nan_to_num(input_mld, nan=0.0)
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
        print(dates)
        train_data_era = nc.Dataset(f'{self.args["data_path"]}/ERA5/025res_{years}.nc')
        #train_data_cmems = nc.Dataset(f'{self.args["data_path"]}/CMEMS/low/CMEMS_low_processed_{years}.nc')
        train_data_glorys = nc.Dataset(f'/jizhicfs/easyluwu/ocean_project/ft_local/ocean_mhws/shuruiqi_code/ft_local/highres/CMEMS_{years}_norm.nc')
        train_data_altimetry = nc.Dataset(f'{self.args["data_path"]}/CMEMS/low/SSH_low_processed_{years}.nc')
        uv_pred_data = nc.Dataset(f'/jizhicfs/easyluwu/NeuralPS_shu/MHWs/uv_pred_{years}.nc')
        train_data_analysis = nc.Dataset(f'/jizhicfs/easyluwu/ocean_project/ft_local/ocean_mhws/shuruiqi_code/ft_local/highres/CMEMS_{years}_norm_analysis.nc')
        train_data_era_of = nc.Dataset(f'{self.args["data_path"]}/ERA5/025res_{years}_of_1s.nc')
        
        input_era = train_data_era.variables['mhws_variables'][dates:dates+self.args['atmosphere_lead_time']+1, 
                                   [2,3,4],
                                   self.args['lat_start']:self.args['lat_end']:self.args['ds_factor'],
                                   self.args['lon_start']:self.args['lon_end']:self.args['ds_factor']]
        # input_era = train_data_era_of.variables['mhws_variables'][dates//3, 
        #                            :, :, 
        #                            self.args['lat_start']:self.args['lat_end'],
        #                            self.args['lon_start']:self.args['lon_end']]
        input_uo = uv_pred_data.variables['pred'][(dates-12)//3, :, 0,
                                    self.args['lat_start']:self.args['lat_end']:self.args['ds_factor'],
                                    self.args['lon_start']:self.args['lon_end']:self.args['ds_factor']]

        input_vo = uv_pred_data.variables['pred'][(dates-12)//3, :, 1,
                                    self.args['lat_start']:self.args['lat_end']:self.args['ds_factor'],
                                    self.args['lon_start']:self.args['lon_end']:self.args['ds_factor']]

        # input_uo = train_data_altimetry.variables['ugos'][dates:dates+self.args['atmosphere_lead_time']+1,
        #                            self.args['lat_start']:self.args['lat_end']:self.args['ds_factor'],
        #                            self.args['lon_start']:self.args['lon_end']:self.args['ds_factor']]

        # input_vo = train_data_altimetry.variables['vgos'][dates:dates+self.args['atmosphere_lead_time']+1,
        #                            self.args['lat_start']:self.args['lat_end']:self.args['ds_factor'],
        #                            self.args['lon_start']:self.args['lon_end']:self.args['ds_factor']]

        #input_mld = train_data_cmems.variables['mld'][dates:dates+self.args['atmosphere_lead_time']+1,
                                   #self.args['lat_start']:self.args['lat_end']:self.args['ds_factor'],
                                   #self.args['lon_start']:self.args['lon_end']:self.args['ds_factor']]
        
        input_ssta = train_data_glorys.variables['ssta_highres'][dates:dates+self.args['ocean_lead_time']+1,
                                   self.args['lat_start']:self.args['lat_end']:self.args['ds_factor'],
                                   self.args['lon_start']:self.args['lon_end']:self.args['ds_factor']]

        # input_ssta = train_data_analysis.variables['ssta_highres'][dates-6:dates+self.args['ocean_lead_time']+1-6,
        #                            self.args['lat_start']:self.args['lat_end'],
        #                            self.args['lon_start']:self.args['lon_end']]
        
        input_era = torch.tensor(input_era.filled())
        input_uo = torch.tensor(input_uo.filled())
        input_vo = torch.tensor(input_vo.filled())
        #input_mld = torch.tensor(input_mld.filled())
        input_ssta = torch.tensor(input_ssta.filled())
        
        input_era = torch.nan_to_num(input_era, nan=0.0)
        input_uo = torch.nan_to_num(input_uo, nan=0.0)
        input_vo = torch.nan_to_num(input_vo, nan=0.0)
        #input_mld = torch.nan_to_num(input_mld, nan=0.0)
        input_ssta = torch.nan_to_num(input_ssta, nan=0.0)
        
        return input_era, input_uo, input_vo, input_ssta

    def __len__(self):
        return len(self.indices)

if __name__ == '__main__':
    args = {
    'data_path': '/jizhicfs/easyluwu/dataset/ft_local',
    'ocean_lead_time': 10,
    'atmosphere_lead_time': 10,
    'shuffle': True,
    'variables_input': [1, 2, 3, 4],
    'variables_future': [2, 3, 4],
    'variables_output': [1],
    'lon_start': 0,
    'lat_start': 0,
    'lon_end': 1440,
    'lat_end': 720,
    'ds_factor': 1,
}


    train_dataset = train_Dataset(args)
    test_dataset = test_Dataset(args)

    train_loader = data.DataLoader(train_dataset, batch_size=2)
    test_loader = data.DataLoader(test_dataset, batch_size=2)

    for inputs, targets in iter(train_loader):
        print(inputs.shape, targets.shape)
        break
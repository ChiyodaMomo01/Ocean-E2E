class Config:
    # 系统配置
    BACKBONE = "mhw_4step_source"
    BACKBONE_PRE = "mhw_4step_source"
    SEED = 42
    CKPT_PATH = f'./checkpoints/{BACKBONE}'
    CKPT_PATH_PRE = f'./checkpoints/{BACKBONE_PRE}'
    BOUNDARY_PATH = '/apdcephfs_qy3/share_301734960/easyluwu/shuruiqi/NeuralPS_25/CMEMS/low/boundary.npy'
    LOG_PATH = "./training.log"
    NUM_WORKERS = 0
    
    # 训练参数
    EPOCHS = 100
    BATCH_SIZE = 1
    INIT_LR = 1e-3
    LR_STEP_SIZE = 10
    LR_GAMMA = 0.2
    NUM_STEPS_TRAIN = 60
    NUM_STEPS_VAL = 60
    
    # 模型参数
    MODEL_GM = {
        'n_channels': 3,
        'n_classes': 2
    }

    MODEL_S = {
        'n_channels': 7,
        'n_classes': 1
    }
    
    # 物理参数
    QGM_PARAMS = {
        'nx': 360,
        'ny': 720,
        'Lx': 4800.e3,
        'Ly': 4800.e3,
        'nl': 3,
        'heights': [350., 750., 2900.],
        'reduced_gravities': [0.025, 0.0125],
        'f0': 9.375e-5,
        'a_2': 0.,
        'a_4': 2.0e9,
        'beta': 1.754e-11,
        'delta_ek': 2.0,
        'dt': 1200.,
        'bcco': 0.2,
        'tau0': 2.e-5,
        'n_ens': 0,
        'device': 'cuda',
        'p_prime': ''
    }
    
    # 数据参数
    DATA = {
        'data_path': '/apdcephfs_qy3/share_301734960/easyluwu/shuruiqi/NeuralPS_25',
        'forecast_path': '/data1/shuruiqi/Projects/MHWs/data/NC_files/Pangu',
        'operational': False,
        'ocean_lead_time': 60,
        'atmosphere_lead_time': 60,
        'shuffle': True,
        'variables_input': [0, 2, 3, 4],
        'variables_future': [2, 3, 4],
        'variables_output': [0],
        'lon_start': 0,
        'lat_start': 0,
        'lon_end': 1440,
        'lat_end': 720,
        'ds_factor':2,
    }

config = Config()
import os
import time
import json
import random
import numpy as np
from tqdm.notebook import tqdm
from joblib import Parallel, delayed

from WJYML.model_trainer import *
from WJYML.data_loader import MyDataset
from WJYML.models.builder import ModelBuilder

def makedir(path_new):
    if not os.path.exists(path_new):
        os.makedirs(path_new)

today = time.strftime('%Y-%m-%d')
logger.info(today)

MODEL_PATH = 'F:/quant_research/model_train/'
DS_PATH = f'{MODEL_PATH}data_store/'
MS_PATH = f'{MODEL_PATH}model_save/'

kfold_info = [5,5]
dataset_name = 'zzfull_alpha158' # 数据地址
model_save_name = f'for2023_5fold_label10_long_fullm_158' # 存储模型文件夹命名
ms_day_path = MS_PATH + today + '/'
MS_ONE = ms_day_path + model_save_name + '/'
makedir(ms_day_path)

train_date_range = '20130101|20230101'
pred_date_range = '20130101|20250101'  
feature_use_list = []
label_use_list = ['LABEL10']
# X_shape = (20,158)
X_shape = (-1,)
data_process_dict = {'features':['cs_zscore','clip_5'],
                     'label':['cs_zscore','clip_5']}

# 模型信息
model_use = 'mlp' 
# model_use = 'lstm'
loss_use = 'ic'
seed_list = [1,2,3,4,5]
total_epoch = 40
batch_size = 3000

# MLP
model_config = dict (
        input_size=158,
        units=[128, 64],
        dropout=0.3,
        need_sigmoid=True,
        is_bn_first=True,
)

# # LSTM
# model_config = dict (
#         input_dim=158,
#         hidden_dim=32,
#         dropout=0.3,
#         need_sigmoid=True,
#         need_encoding=True,
# )

# 运行信息
run_config = dict(
    today = today,

    dataset_name = dataset_name, # 数据地址
    model_save_name = model_save_name, # 存储模型文件夹命名

    train_date_range = train_date_range,
    pred_date_range = pred_date_range,   
    kfold_info = kfold_info,
    feature_use_list = feature_use_list,
    label_use_list = label_use_list,
    X_shape = X_shape,
    data_process_dict = data_process_dict,

    model_use = model_use,
    loss_use = loss_use,
    seed_list = seed_list,
    total_epoch = total_epoch,
    batch_size = batch_size,
    model_config = model_config,

    DS_PATH = DS_PATH,
    MS_PATH = MS_PATH,
    MS_ONE = MS_ONE,
)

if __name__ == '__main__':
    start_time = time.strftime('%Y-%m-%d %H:%M:%S')
    logger.info(run_config)
    train_s, train_e = train_date_range.split('|')
    pred_s, pred_e = pred_date_range.split('|')
    
    model_dict = ModelBuilder(model_use,
                            seed_list,
                            model_config).model_dict_generate()

    train_ds = MyDataset(DS_PATH, dataset_name, train_s, train_e, 
                        feature_use_list = feature_use_list, label_use_list = label_use_list,
                        X_shape = X_shape, data_process_dict = data_process_dict)
    # pred_ds = MyDataset(DS_PATH, dataset_name, pred_s, pred_e, 
    #                     feature_use_list = feature_use_list, label_use_list = label_use_list,
    #                     X_shape = X_shape, data_process_dict = data_process_dict)

    nn_run = WJYNN(model_dict)
    nn_run.train(train_ds, batch_size=batch_size, epoch_num=total_epoch,
                 kfold_info=kfold_info,
                 num_workers=2, criterion_kind=loss_use)

    # 存模型
    nn_run.save_models(MS_ONE)
    # 存train_dataset的输出值和相关信息
    nn_run.save_output(MS_ONE, train_ds, num_workers=2,batch_size=batch_size,)
    # 如果训练成功，将字典写入JSON文件
    with open(MS_ONE + 'run_map.json', 'w') as f:
        json.dump(run_config, f)
    logger.info(run_config)

    # 记录时间
    end_time = time.strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"start time: {start_time} end time: {end_time} ALL FINISH!")

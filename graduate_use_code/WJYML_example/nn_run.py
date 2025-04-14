import os
import time
import json
import random
import numpy as np
import threading
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

file_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在目录，或者直接改MODEL_PATH
MODEL_PATH = f'{file_path}/model_train/'
DS_PATH = f'{MODEL_PATH}data_store/'
MS_PATH = f'{MODEL_PATH}model_save/'

kfold_info = [5,5] # [分几折，第几折作为为validation集合], [5,5]意为分8：2，后20%不参加训练
dataset_name = 'example_alpha158_rzscore_n' # 数据地址
model_save_name = f'for2025_5fold_label20_long_zzfull_158_icloss_rzscore_n_sigmoid_diff' # 存储模型文件夹命名
ms_day_path = MS_PATH + today + '/'
MS_ONE = ms_day_path + model_save_name + '/'
makedir(ms_day_path)

train_date_range = '20240101|20250101' # 样本内
pred_date_range = '20240101|20260101'  # 样本内+样本外
feature_use_list = [] # 可以指定用哪些特征，空list意为用所有的
label_use_list = ['LABEL20'] # 可以指定用哪些label
# X_shape = (20,158)
X_shape = (-1,) 
# 如果不确定样本的X_shape，默认设置为(-1,)，即使用feature的.npy数据作为一列；.npy中一个样本必为一列，如果要用多维度，就在这里调整
data_process_dict = {'features':['clip_10'], # 'cs_zscore',
                     'label':['cs_rank','clip_5']}

# 模型信息
model_use = 'mlp' 
# model_use = 'lstm'
loss_use = 'ic'
seed_list = [1,2,3,4,5]
total_epoch = 30
batch_size = 3000
learning_rate = 5e-6
batch_bydate = True # 如果为true，优先级高于batch_sizes，batch_size为每个日期的样本数

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

# 记录运行信息（以下就没有可调整的部分了）
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
    learning_rate = learning_rate,
    batch_bydate = batch_bydate,
    model_config = model_config,

    DS_PATH = DS_PATH,
    MS_PATH = MS_PATH,
    MS_ONE = MS_ONE,
)

if __name__ == '__main__':

    thread = threading.Thread(target=monitor_memory)
    thread.daemon = True  # 设置为守护线程，这样主线程结束时线程也会结束
    thread.start()

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
    
    nn_run = WJYNN(model_dict)
    nn_run.train(train_ds, batch_size=batch_size, epoch_num=total_epoch,learning_rate = learning_rate,
                 batch_bydate = batch_bydate, kfold_info=kfold_info,num_workers=4, criterion_kind=loss_use)

    # 存模型
    nn_run.save_models(MS_ONE)
    # 存train_dataset的输出值和相关信息
    nn_run.save_output(MS_ONE, train_ds, num_workers=4, batch_size=batch_size,)
    # 如果训练成功，将字典写入JSON文件
    with open(MS_ONE + 'run_map.json', 'w') as f:
        json.dump(run_config, f)
    logger.info(run_config)

    # 停止资源监控子线程
    threading.Event().set()
    # 记录时间
    end_time = time.strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"start time: {start_time} end time: {end_time} ALL FINISH!")
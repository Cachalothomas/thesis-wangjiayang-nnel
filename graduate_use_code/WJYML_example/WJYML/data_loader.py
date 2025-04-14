import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import os
import gc
import importlib.util

import torch
from torch.utils.data import Dataset

# logger预设
from WJYML.base_logger import logger

import warnings
# 忽略所有的FutureWarning(在groupby的时候遇到)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

class MyDataset(Dataset):
    def __init__(self, DS_PATH, dataset_name, 
                 start_date, end_date, X_shape: tuple=(-1,),
                 feature_use_list: list = [], # 空集默认取全部feature
                 label_use_list: list = [], # 空集默认取第一个label, 因为暂时还不支持多目标训练
                 data_process_dict: dict = {'features':[],
                                            'label':[]}):
        super().__init__()

        self.DS_PATH = DS_PATH 
        self.dataset_name = dataset_name
        self.X_shape = X_shape
        self.feature_use_list = feature_use_list
        self.label_use_list = label_use_list
        
        self.data_process_dict = data_process_dict

        # import对应的信息数据
        info_path = self.DS_PATH + self.dataset_name +"/feature_label_info.py"
        info = ImportVar(info_path)
        self.feature_list = info.feature_list if len(feature_use_list) == 0 else feature_use_list
        self.label_list = [info.label_list[0]] if len(label_use_list) == 0 else label_use_list
        # 找到目标feature和label的位置
        self.feature_loc = self.find_loc(info.feature_list, self.feature_list)
        self.label_loc = self.find_loc(info.label_list, self.label_list)

        info.all_date_list.sort()
        self.all_date_list = np.array(info.all_date_list)
        
        self.target_date_list = self.all_date_list[(self.all_date_list>=start_date) \
                                                   & (self.all_date_list<end_date)]
        assert len(self.target_date_list)!=0
        self.readin_all()
        self.data_process(self.data_process_dict)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = torch.as_tensor(self.features[idx], dtype=torch.float32)   
        y = torch.as_tensor(self.label[idx], dtype=torch.float32)  
        return x, y

    def find_loc(self, main_list: list, subset: list) -> list:
        '''寻找list子集元素在原list中的位置, 主要为了实现从ds_one中只用部分数据训练'''
        loc_dict = {value: key for key, value in dict(enumerate(main_list)).items()}
        return [loc_dict[i] for i in subset]

    def oneday_readin(self, date_str,):
        X_tmp = np.load(self.DS_PATH + self.dataset_name + f"/feature/{date_str}.npy")[:,self.feature_loc]
        y_tmp = np.load(self.DS_PATH + self.dataset_name + f"/label/{date_str}.npy")[:,self.label_loc]
        idx_tmp = np.load(self.DS_PATH + self.dataset_name + f"/index/{date_str}.npy",allow_pickle=True)

        feature_dict = dict(zip(idx_tmp,X_tmp))
        label_dict = dict(zip(idx_tmp,y_tmp))
        return feature_dict, label_dict
    
    def readin_all(self,):
        target_data_list = Parallel(n_jobs=100,backend='threading',verbose=0)\
            (delayed(self.oneday_readin)(date_str,)for date_str in self.target_date_list)

        feature_dict = {}
        label_dict = {}
        for i in target_data_list:
            feature_dict.update(i[0])
            label_dict.update(i[1])

        assert list(feature_dict.keys()) == list(label_dict.keys())
        del target_data_list
        gc.collect()
        # pd.DataFrame.from_dict(feature_dict, orient='index',columns=feature_list)
        # pd.DataFrame.from_dict(label_dict, orient='index',columns=label_list)
        self.features = np.array(list(map(lambda x: x.reshape(self.X_shape),feature_dict.values())))
        self.label = np.array(list(label_dict.values()))
        self.idx_dict = dict(list(enumerate(feature_dict.keys())))

        self.idx_list = np.array(list(self.idx_dict.values()),dtype='object')
        self.idx_sep_list = np.array(list(map(lambda x: x.split('|'),self.idx_list)),dtype='object')

        self.X_shape = self.features.shape[1:]

        del feature_dict
        gc.collect()
        del label_dict
        gc.collect()

    def ary2df(self, process_kind: str):
        '''把ary转df 用以data预处理'''
        if process_kind == 'features':
            data_reshape = self.features.reshape(self.features.shape[0],-1)
            # del self.features
            # gc.collect()
        elif process_kind == 'label':
            data_reshape = self.label.reshape(self.label.shape[0],-1)
            # del self.label
            # gc.collect()
        else:
            raise ValueError(f'process_kind |{process_kind}| wrong')
        df_preprocess = pd.DataFrame(data_reshape,index=self.idx_list)
        df_preprocess[['datetime','instrument']] = self.idx_sep_list
        return df_preprocess

    def df2ary(self, df_done: pd.DataFrame, process_kind: str):
        '''把df转回self.values'''

        data_done = df_done.loc[self.idx_list,:].drop(columns=['datetime','instrument']).values
        data_done = data_done.astype(np.float16)
        if process_kind == 'features':
            self.features = data_done.reshape(self.features.shape)
        elif process_kind == 'label':
            self.label = data_done.reshape(self.label.shape)
        else:
            raise ValueError(f'process_kind |{process_kind}| wrong')
        del df_done
        gc.collect()

    def process_acc(self, x, func):
        '''使用threading对处理函数加速, 比groupby apply快多了'''
        res = Parallel(n_jobs=100,backend='threading',verbose=0)(delayed(func)\
                                                             (g[1]) for g in x.groupby(by='datetime'))
        return pd.concat(res,axis=0)
    
    def data_process(self,
                     data_process_dict: dict = {'features':[],
                                                'label':[]}):
        cl = 2 # 默认np.clip的上下限是2
        process_func_dict = {
                    'cs_zscore': lambda x: self.process_acc(x,func=self.cs_zscore_base),
                    'cs_minmax': lambda x: self.process_acc(x,func=self.cs_minmax_base), 
                    'cs_rank': lambda x: self.process_acc(x,func=self.cs_rank_base),
                    }
                            # 'clip': lambda x: np.clip(x.astype(float),-float(cl),float(cl))}

        for k in data_process_dict:
            process_kind = k
            process_list = data_process_dict[k]
            if len(process_list) == 0:
                continue
            else:
                df_process = self.ary2df(process_kind)
                need_clip = False
                for item in process_list:
                    logger.info(f"{process_kind} {item} processing...")
                    if 'clip' in item: # 如果是clip 要把clip_limit提取出来
                        item, cl = item.split('_')
                        need_clip = True
                        continue # 跳过去

                    df_process = process_func_dict[item](df_process) # 一定要注意计算完后新出的异常值
                    df_process[['datetime','instrument']] = self.idx_sep_list
                    logger.info(f"{process_kind} {item} done")

                self.df2ary(df_process, process_kind)

                del df_process
                gc.collect()

                if need_clip:
                    need_clip = False
                    if process_kind == 'features':
                        self.features = np.clip(self.features,-float(cl),float(cl))
                    elif process_kind == 'label':
                        self.label = np.clip(self.label,-float(cl),float(cl))
                    else:
                        raise ValueError(f'process_kind |{process_kind}| wrong')
                    logger.info(f"{process_kind} {item} done")
                
        logger.info('process done!')

    def cs_rank_base(self, df_tmp:pd.DataFrame):
        df_tmp = df_tmp[[c for c in df_tmp.columns if type(c)==int]]
        df_tmp = df_tmp.rank(axis=0,pct=True)
        return df_tmp - 0.5

    def cs_minmax_base(self, df_tmp:pd.DataFrame):
        df_tmp = df_tmp[[c for c in df_tmp.columns if type(c)==int]]
        df_tmp = (df_tmp - df_tmp.min(axis=0)).divide(df_tmp.max(axis=0) - df_tmp.min(axis=0),axis=1) 
        df_tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_tmp.fillna(0.5,inplace=True)
        return df_tmp - 0.5

    def cs_zscore_base(self, df_tmp:pd.DataFrame):
        df_tmp = df_tmp[[c for c in df_tmp.columns if type(c)==int]]
        df_tmp = (df_tmp - df_tmp.mean(axis=0)).divide(df_tmp.std(axis=0),axis=1) 
        df_tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_tmp.fillna(0.,inplace=True)
        return df_tmp 

# import py文件中的变量
class ImportVar:
    def __init__(self, file_path):
        # 确保文件存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        # 导入模块
        spec = importlib.util.spec_from_file_location("module.name", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 将模块中的变量作为类属性
        for attribute_name in dir(module):
            # 只复制模块中的变量，不复制内置属性和方法
            if not attribute_name.startswith('__'):
                setattr(self, attribute_name, getattr(module, attribute_name))
import os
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from WJYML.data_loader import MyDataset

from tqdm import tqdm
from joblib import Parallel, delayed
from pathlib import Path
from copy import deepcopy

# loss_func
from WJYML.loss_func import NNLossFunc 
# logger预设
from WJYML.base_logger import *
# 存储模型用
from WJYML.torch_convertor import save_torch_in_pth #save_torch_in_onnx,

import warnings
warnings.filterwarnings("ignore")

class WJYNN():
    def __init__(
        self,
        models: dict,
        # model_name: str,
        # model_num: int,
        # model_param: dict,
    ):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.seed_list = list(models.keys())
        self.models = list(models.values())
        for model in self.models:
            model.to(self.device)

        self.LOSS_FUNC = NNLossFunc()

    def makedir(self,path_new):
        if not os.path.exists(path_new):
            os.makedirs(path_new)

    def setup_seed(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def train(
        self,
        dataset: MyDataset,
        num_workers=0,
        kfold_info: list=[1,1], # 是否需要划分kfold, 默认[1,1]就是分一组(不分组)全用来训练
        learning_rate=1e-3,
        batch_size=2048,
        epoch_num=1,
        criterion_kind='None',
        batch_bydate: bool=True,
        **kwargs,
    ):
        
        # 设置kfold
        total_group, mask_num = kfold_info
        date_list = np.unique(np.sort(dataset.idx_sep_list[:,0]))

        mask_date = np.array_split(date_list,total_group)[mask_num-1]
        df_idx = pd.DataFrame(dataset.idx_sep_list,columns=['datetime','instrument'])
        df_idx_use = df_idx.loc[~df_idx['datetime'].isin(mask_date)] if total_group != 1 else df_idx
        train_idx = df_idx_use.index.values 

        dataset_used = torch.utils.data.dataset.Subset(dataset, train_idx)
        df_idx_use['subset_idx'] = range(len(df_idx_use))

        # 根据日期分batch的准备: 将分组结果转换为字典
        grouped = df_idx_use.groupby(by='datetime')
        grouped_dict = {name: group['subset_idx'].tolist() for name, group in grouped}

        device=self.device
        g = torch.Generator()
        # 设置随机数种子
        g.manual_seed(0)
        # s = CustomSampler(dataset_used)
        if batch_bydate:
            date_bs = CustomBatchSampler(grouped_dict)
            dataloader = DataLoader(
                dataset_used,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True, # 防止epoch之间再load一次了
                generator=g,
                batch_sampler=date_bs,
            )
        else:
            dataloader = DataLoader(
                dataset_used,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
                persistent_workers=True, # 防止epoch之间再load一次了
                generator=g,
                )

        self.all_models = {}
        optimizers = []
        # 设置优化器
        for model in self.models:
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            optimizers.append(optimizer)
        # 设置损失函数
        if criterion_kind == 'None':
            # criterion = self.criterion_ic
            criterion = self.LOSS_FUNC.criterion_ic
        else:
            criterion = self.LOSS_FUNC.func_dict[criterion_kind]
        # 开始训练
        for i_epoch in range(epoch_num):
            epoch_loss_list = []
            for i_batch, (feature, label) in enumerate(dataloader):
                feature = feature.to(device)
                label = label.to(device)

                loss_doc = []
                for i_model in range(len(self.models)):
                    model = self.models[i_model]
                    optimizer = optimizers[i_model]
                    random_seed = self.seed_list[i_model]
                    # label剔除掉nan, feature的nan值用0填充
                    self.setup_seed(random_seed)

                    train_sample = ~torch.isnan(label).reshape(-1,)
                    train_feature = feature[train_sample]
                    train_feature = torch.nan_to_num(train_feature)
                    train_label = label[train_sample]

                    pred = model(train_feature)
                    loss = criterion(pred.reshape(-1), train_label.reshape(-1))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_doc.append(loss.item())
                epoch_loss_list.append(np.mean(loss_doc))

                if i_batch % 500 == 0:
                    logger.info(
                        f"training {i_epoch+1} epoch {i_batch} batches,"
                        + f"loss mean {round(np.mean(loss_doc),4)}\n" 
                        + f"{[round(_, 8) for _ in loss_doc]}"
                    )

            logger.info(
                f"training {i_epoch+1} epoch [mean epoch loss] {round(np.mean(epoch_loss_list), 8)}"
            )

            # 每轮训完保存模型
            self.all_models[f"epoch{i_epoch+1}"] = [deepcopy(_) for _ in self.models]
    
    def pred(
        self,
        dataset: Dataset,
        num_workers=0,
        check_epoch=1,
        batch_size=2048,
        saving_all_epoch_res=True,
        **kwargs,
    ):
        
        # TODO: batch_size 需要与train保持一致
        device=self.device
        self.check_epoch = check_epoch
        self.saving_all_epoch_res = saving_all_epoch_res

        g = torch.Generator()
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            generator=g,
        )

        if self.saving_all_epoch_res:
            save_epoch = list(self.all_models.keys())
        else:
            save_epoch = [f"epoch{self.check_epoch}"]

        preds = {epoch: [[] for _ in range(len(self.models))] for epoch in save_epoch}
        
        for (feature, label) in tqdm(dataloader):
            feature = feature.to(device)
            for epoch, epoch_preds in preds.items():
                for i_model in range(len(self.models)):
                    model = self.all_models[epoch][i_model]
                    model.eval()
                    pred = model(feature)
                    epoch_preds[i_model].append(pred.reshape(-1).detach().cpu().numpy())

        for epoch in preds.keys():
            preds[epoch] = [  # type:ignore
                np.concatenate(_, axis=0) for _ in preds[epoch]
            ]  # type:ignore

        return preds
    
    def save_models(self, save_path,):
        self.makedir(save_path)
        for epoch, models in self.all_models.items():
            for i_model in range(len(models)):
                model = models[i_model]
                random_seed = self.seed_list[i_model]
                model_path = Path(save_path,
                        f"model_{epoch}_seed{random_seed}",)
                with torch.no_grad():
                    save_torch_in_pth(model, model_path)

    def load_models(self, save_path,):
        dirlist = os.listdir(save_path) 
        dirlist = [d for d in dirlist if 'pth' in d]
        epoch_list = [d.split('.')[0].split('_')[1] for d in dirlist]
        
        new_all_models = dict()
        load_dict = {}
        for i,e in enumerate(epoch_list):
            if e in load_dict:
                load_dict[e] += [dirlist[i]]
            else:
                load_dict[e] = [dirlist[i]]


        for epoch, models_name_list in load_dict.items():
            models_list = []
            for model_name in models_name_list:
                pth_path = save_path + model_name
                model = torch.load(
                    pth_path, map_location=torch.device(self.device)
                )
                models_list.append(model)
            # print(epoch, models_list)
            new_all_models[epoch] = models_list
        self.all_models = new_all_models

    def save_output(
        self, 
        save_path,
        dataset: Dataset,
        num_workers=0,
        check_epoch=1,
        batch_size=2048,
        saving_all_epoch_res=True,
        **kwargs,
    ):

        self.makedir(save_path)
        # 存train_dataset的输出值
        train_ouput = self.pred(dataset,
                                num_workers=num_workers,
                                check_epoch=check_epoch,
                                batch_size=batch_size,
                                saving_all_epoch_res=saving_all_epoch_res,)
        train_epoch_list = list(train_ouput.keys())
        train_ouput_values = np.array(list(train_ouput.values())).astype(np.float16)
        np.save(save_path + 'train_output.npy', train_ouput_values)
        np.save(save_path + 'train_label.npy', dataset.label)
        
        # 存train_dataset的信息
        output_info_dict = dict(train_epoch_list = train_epoch_list,
                                idx_sep_list = dataset.idx_sep_list.tolist(),)
        with open(save_path + 'train_output.json', 'w') as f:
            json.dump(output_info_dict, f)

# 实现自定义batch划分方式
class CustomBatchSampler:
    def __init__(self, grouped_dict: dict,):
        self.grouped_dict = grouped_dict

    def __iter__(self):
        batch = []
        for k in self.grouped_dict.keys():
            batch = self.grouped_dict[k]
            yield batch

    def __len__(self):
        return len(self.grouped_dict)
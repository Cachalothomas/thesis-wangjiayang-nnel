import torch
import numpy as np
import random

from .mlp import MLP
from .lstm import LSTM_Model

class ModelBuilder:
    def __init__(self,
                 model_use,
                 seed_list,
                 model_config):

        self.model_kind_dict = {
            'mlp': MLP,
            'lstm': LSTM_Model,
        }

        self.model_use = self.model_kind_dict[model_use]
        self.seed_list = seed_list
        self.model_config = model_config
    
    def model_dict_generate(self,):
        model_dict = {}
        for random_seed in self.seed_list:
            self.setup_seed(random_seed)
            model = self.model_use(**self.model_config)
            model_dict[random_seed] = model

        # 展示一个模型的结构
        self.model_structure(list(model_dict.values())[0])

        return model_dict

    def setup_seed(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    
    def model_structure(self,model):
        blank = ' '
        print('-' * 90)
        print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
            + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
            + ' ' * 3 + 'number' + ' ' * 3 + '|')
        print('-' * 90)
        num_para = 0
        type_size = 1  # 如果是浮点数就是4
        outputlist = []
        for index, (key, w_variable) in enumerate(model.named_parameters()):
            if len(key) <= 30:
                key = key + (30 - len(key)) * blank
            shape = str(w_variable.shape)
            if len(shape) <= 40:
                shape = shape + (40 - len(shape)) * blank
            each_para = 1
            for k in w_variable.shape:
                each_para *= k
            num_para += each_para
            str_num = str(each_para)
            if len(str_num) <= 10:
                str_num = str_num + (10 - len(str_num)) * blank

            print('| {} | {} | {} |'.format(key, shape, str_num))
            outputlist.append([key, shape, str_num])
        print('-' * 90)
        print('The total number of parameters: ' + str(num_para))
        print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
        print('-' * 90)
        return outputlist

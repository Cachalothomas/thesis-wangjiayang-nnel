import torch 
import torch.nn as nn

'''
nn.modules.loss.__all__ = ['L1Loss', 'NLLLoss', 'NLLLoss2d', 'PoissonNLLLoss', 'GaussianNLLLoss', 'KLDivLoss', 'MSELoss', 'BCELoss', 'BCEWithLogitsLoss', 'HingeEmbeddingLoss', 'MultiLabelMarginLoss', 'SmoothL1Loss', 
'HuberLoss', 'SoftMarginLoss', 'CrossEntropyLoss', 'MultiLabelSoftMarginLoss', 'CosineEmbeddingLoss', 'MarginRankingLoss', 'MultiMarginLoss', 'TripletMarginLoss', 'TripletMarginWithDistanceLoss', 'CTCLoss']
'''

class NNLossFunc():
    def __init__(self) -> None:
        self.func_dict = {
            'ic':self.criterion_ic,
            'mse':self.criterion_mse,
            'l1':self.criterion_l1,
            'icmse':self.criterion_icmse,
        }
        pass

    def criterion_icmse(self, y_pred, y_true):
        w = 0.1
        y_pred = y_pred.reshape(-1,1)
        y_true = y_true.reshape(-1,1)
        rtn_corr = torch.corrcoef(torch.concat([y_pred,y_true],axis=1).T)[0][1]
        # y_pred = y_pred - torch.mean(y_pred).item() + torch.mean(y_true).item() # 统一中心均值
        return -rtn_corr + w * nn.MSELoss(reduction='mean')(y_pred,y_true)

    def criterion_ic(self, y_pred, y_true):
        y_pred = y_pred.reshape(-1,1)
        y_true = y_true.reshape(-1,1)
        rtn_corr = torch.corrcoef(torch.concat([y_pred,y_true],axis=1).T)[0][1]
        return -rtn_corr
    
    def criterion_mse(self, y_pred, y_true):
        y_pred = y_pred.reshape(-1,1)
        y_true = y_true.reshape(-1,1)
        # y_pred = y_pred - torch.mean(y_pred).item() + torch.mean(y_true).item() # 统一中心均值
        return nn.MSELoss(reduction='mean')(y_pred,y_true)

    def criterion_l1(self, y_pred, y_true):
        y_pred = y_pred.reshape(-1,1)
        y_true = y_true.reshape(-1,1)
        # y_pred = y_pred - torch.mean(y_pred).item() + torch.mean(y_true).item() # 统一中心均值
        return nn.L1Loss(reduction='mean')(y_pred,y_true)
        # y_pred = torch.rand(10)
        # y_true = torch.rand(10)
        # nn.L1Loss(reduction='mean')(y_pred, y_true)


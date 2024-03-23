# -*- coding: utf-8 -*-
"""
    Created on 2022/11/12 00:28
    @author: wjy
"""
import os, time, datetime, gc
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm 
import warnings
warnings.filterwarnings('ignore')

# 压缩df函数
def reduce_mem_usage(df):
    starttime = time.time()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.isnull(c_min) or pd.isnull(c_max):
                continue
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float)
    end_mem = df.memory_usage().sum() / 1024**2
    print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min'.format(end_mem,
                                                                                                           100*(start_mem-end_mem)/start_mem,
                                                                                                           (time.time()-starttime)/60))
    return df

def factor_cal(stk):
    stk.index = stk['date']
    stk['ret'] = stk['close'] / stk['open'] - 1 
    stk['range'] = stk['high'] / stk['low'] - 1 
    stk['night_jump'] = stk['open'] / stk['close'].shift(1) - 1 
    stk['label'] = stk['open'].shift(-2) / stk['open'].shift(-1) - 1
    # stk['label'] = stk['close'].shift(-5) / stk['open'].shift(-1) - 1
    stk['label'][stk['low'].shift(-1) == stk['high'].shift(-1)] = np.nan

    result = stk.loc[:,['date','instrument','label']]

    # 简单滞后
    single = ['turn','amount','open','close','high','low','ret','range','night_jump']
    for i in range(5):
        columns = [x+'_'+str(i) for x in single]
        result[columns] = stk[single].shift(i)

    # 区间求和、最大值、最小值、标准差
    window = 5
    # item = ['mean','ts_max','ts_min','std','ts_rank']

    columns = ['mean'+'_'+str(window)+x for x in single]
    result[columns] = stk[single].rolling(window=window).mean()

    columns = ['ts_max'+'_'+str(window)+x for x in single]
    result[columns] = stk[single].rolling(window=window).max()

    columns = ['ts_min'+'_'+str(window)+x for x in single]
    result[columns] = stk[single].rolling(window=window).min()

    columns = ['std'+'_'+str(window)+x for x in single]
    result[columns] = stk[single].rolling(window=window).std()

    def ts_rank(a):
        sorted_id = sorted(range(len(a)), key=lambda k: a[k], reverse=True)
        return sorted_id[-1]

    columns = ['ts_rank'+'_'+str(window)+x for x in single]
    result[columns] = stk[single].rolling(window=window).apply(ts_rank)

    # 相关性类
    item = [
        ['volume','ret'],
        ['volume','open'],
        ['volume','high'],
        ['volume','low'],
        ['volume','close'],
        ['volume','turn'],
        ['volume','range'],
        ['volume','night_jump'],

        ['ret','open'],
        ['ret','high'],
        ['ret','low'],
        ['ret','close'],
        ['ret','turn'],
        ['ret','range'],
        ['ret','night_jump'],

        ['high','low'],
        ['high','open'],
        ['high','close'],
        ['high','turn'],
        ['high','range'],
        ['high','night_jump'],

        ['low','open'],
        ['low','close'],
        ['low','turn'],
        ['low','range'],
        ['low','night_jump'],

        ['open','close'],
        ['open','turn'],
        ['open','range'],
        ['open','night_jump'],
        
        ['turn','close'],
        ['turn','range'],
        ['turn','night_jump'],
        
        ['close','range'],
        ['close','night_jump'],
        
        ['night_jump','range']
    ]

    for i in item:
        result['corr_'+i[0]+'_'+i[1]+'_'+str(window)] = stk[i[0]].rolling(window=window).corr(stk[i[1]])
    del stk
    gc.collect()
    return result


def standardlize(one_day):
    one_day = one_day.replace([np.inf,-np.inf],np.nan)
    one_day = one_day[(one_day['turn_0']!=0)&(one_day['turn_0'].notnull())&(one_day['label'].notnull())]
    one_day = one_day.sort_values(by='label')
    number = one_day.iloc[:,2:]
    # number = number * np.array((number['corr_turn_close_5']/number['corr_turn_close_5'])).reshape(-1,1)
    number = (number - np.mean(number)) / np.std(number)

    # 分组操作
    n = 10
    label_check = number['label'] + np.random.normal(0, 1, size=(len(number))) * 1e-12
    group = pd.qcut(label_check,n,labels=range(n)).to_frame('group')
    number = pd.concat([group,number],axis=1)

    base = one_day.iloc[:,:3].rename(columns={'label':'label_raw'})
    one_day = pd.concat([base,number.astype(np.float16)],axis=1)
    one_day = one_day.sort_values(by='instrument')
    return one_day

if __name__ == '__main__':
    path = os.path.dirname(os.path.realpath(__file__))
    print(path)

    # print("\n{:-^120s}".format('因子生成'))
    # print('[{}]处理中……'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
    # # df = pd.read_pickle(path + '\Bar_1d_raw.pkl')
    # # df = pd.read_pickle('./ bar_1d_raw.pkl')

    # df1 = pd.read_parquet('D:\jupyter notebook\my_invest\stockdata_all.parquet')
    # df = df1.rename(columns={'turnover':'turn','id':'instrument'})

    # group = []
    # for i in df.groupby(by='instrument'):
    #     group.append(i[1]) 
    
    # print('[{}]处理完毕，计算中……'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))

    # # 使用multiprocessing并行计算
    # with Pool() as pool:
    #     result = list(tqdm(pool.imap(factor_cal, group), total=len(group)))

    # print('[{}]计算完毕，合并中……'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))    

    # factor_raw = pd.concat(result,axis=0)
    # factor_raw.index = range(len(factor_raw))
    # factor_raw = reduce_mem_usage(factor_raw)
    factor_raw = pd.read_pickle(path + '\Factor_raw.pkl')

    print("\n{:-^120s}".format('标准化'))
    print('[{}]处理中……'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
    factor_raw = factor_raw[factor_raw['date']<='2023-12-15'] # 最后几天的label基本是nan，必须去掉防止分组报错
    group = []
    for i in factor_raw.groupby(by='date'):
        group.append(i[1]) 
    
    print('[{}]处理完毕，计算中……'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))

    # 使用multiprocessing并行计算
    with Pool() as pool:
        result = list(tqdm(pool.imap(standardlize, group), total=len(group)))

    print('[{}]计算完毕，合并中……'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))) 

    factor_final = pd.concat(result,axis=0)
    factor_final.index = range(len(factor_final))
    
    print('[{}]合并完毕，储存中……'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))  
    factor_final.to_pickle(path + '\Factor_final.pkl')


    print('全部运行完毕！')




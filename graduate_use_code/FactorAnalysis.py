import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import scipy.stats
from datetime import datetime
from tqdm.notebook import tqdm
from joblib import Parallel, delayed
# 图画面板调整为白色
rc = {'axes.facecolor': 'white',
      'savefig.facecolor': 'white'}
mpl.rcParams.update(rc)
# 显示负号
mpl.rcParams['axes.unicode_minus']=False

# 显示中文黑体
mpl.rcParams['font.sans-serif'] = ['SimHei']

# 提高照片清晰度 dpi:dots per inch
mpl.rcParams['figure.dpi'] = 200

# 设置样式
plt.style.use('_mpl-gallery')


class FactorAnalysis:
    def __init__(self, factor, ret, list_periods, n_group,benchmark = False, cost = 3e-4, ret_kind = 'sum'):
        self.factor = factor
        self.ret = ret
        self.list_periods = list_periods
        self.n_group = n_group
        self.rf = 0
        self.cost = cost
        self.benchmark = benchmark
        self.ret_kind = ret_kind

        self.factor.index = pd.to_datetime(self.factor.index)
        self.ret.index = pd.to_datetime(self.ret.index)
    
    # ==============================
    # ==============================
    # 计算return，ic，rank_ic，换手率
    # ==============================
    # ==============================
    def calc_results(self, adj_periods):
        n_group = self.n_group
        step = adj_periods
        w = self.cost
        def one_transaction(idx_n,step):
            date = self.ret.index[idx_n]
            ret = self.ret.iloc[idx_n:idx_n+step,:].T
            factor = self.factor.iloc[idx_n,:].to_frame('factor')
            tmp = pd.concat([factor,ret],axis=1)
            # 根据factor的值去掉nan，没有用到未来的股价信息
            tmp.dropna(subset=['factor'], inplace=True)
            tmp.dropna(subset=[tmp.columns[1]], inplace=True)
            tmp['ret'] = (1+tmp.iloc[:,1:]).prod(axis=1) -1 
            tmp = tmp[['factor','ret']]
            tmp['factor'] += np.random.normal(0, 1, size=(tmp.shape[0])) * 1e-12 # 增加一个随机扰动
            tmp['group'] = pd.qcut(tmp['factor'], n_group, labels=list(range(1, n_group+1)))
            return date, tmp

        def get_ret_dict(tmp_pre,tmp_now,tmp_next):
            ret_dict = {}
            for g in range(1,n_group+1):
                one_group_long = tmp_now[tmp_now['group']==g].copy()
                one_group_short = tmp_now[tmp_now['group']==g].copy()
                one_group_short['ret'] = -one_group_short['ret'] 
                out = 0
                for i in one_group_long.index:
                    if i not in tmp_pre[tmp_pre['group']==1].index:
                        one_group_long.loc[i,'ret'] -= w # 开仓费用
                        one_group_short.loc[i,'ret'] -= w # 开仓费用
                    elif i not in tmp_next[tmp_next['group']==1].index:
                        one_group_long.loc[i,'ret'] -= w # 平仓费用
                        one_group_short.loc[i,'ret'] -= w # 平仓费用
                        out += 1
                    else:
                        pass
                long_ret = np.mean(one_group_long['ret'])
                short_ret = np.mean(one_group_short['ret'])

                exchange = out / max(len(tmp_now[tmp_now['group']==g]),1)
                # factor_mean = np.mean(one_group_long['factor'])
                group_ret = {'long':long_ret,'short':short_ret,'exchange':exchange}
                ret_dict['group_'+str(g)] = group_ret
            return ret_dict

        # head
        idx_n = 0
        date_now, tmp_now = one_transaction(idx_n,step)
        date_next, tmp_next = one_transaction(idx_n+step,step)

        ic = tmp_now[['factor','ret']].corr(method='pearson').iloc[0, 1]
        rank_ic = tmp_now[['factor','ret']].corr(method='spearman').iloc[0, 1]
        head_dict = {'date':date_now,'ic':ic,'rank_ic':rank_ic}

        ret_dict = get_ret_dict(tmp_now,tmp_now,tmp_next)
        head_dict = dict(head_dict,**ret_dict)

        # middle
        def get_middle(idx_n):
            date_pre, tmp_pre = one_transaction(idx_n-step,step)
            date_now, tmp_now = one_transaction(idx_n,step)
            date_next, tmp_next = one_transaction(idx_n+step,step)

            ic = tmp_now[['factor','ret']].corr(method='pearson').iloc[0, 1]
            rank_ic = tmp_now[['factor','ret']].corr(method='spearman').iloc[0, 1]

            middle_dict = {'date':date_now,'ic':ic,'rank_ic':rank_ic}
            ret_dict = get_ret_dict(tmp_pre,tmp_now,tmp_next)
            middle_dict = dict(middle_dict,**ret_dict)
            return middle_dict

        # tail
        def get_tail(idx_n):
            date_now, tmp_now = one_transaction(idx_n,step)
            date_pre, tmp_pre = one_transaction(idx_n-step,step)

            ic = tmp_now[['factor','ret']].corr(method='pearson').iloc[0, 1]
            rank_ic = tmp_now[['factor','ret']].corr(method='spearman').iloc[0, 1]
            tail_dict = {'date':date_now,'ic':ic,'rank_ic':rank_ic}

            ret_dict = get_ret_dict(tmp_pre,tmp_now,tmp_now)
            tail_dict = dict(tail_dict,**ret_dict)
            return tail_dict

        if len(self.factor)%step == 0:
            middle_list = Parallel(n_jobs = -1, verbose = 0)(delayed(get_middle)(i*step) 
                                                                for i in tqdm(range(1,len(self.factor)//step-1)))
            tail_dict = get_tail((len(self.factor)//step -1)* step)
        else:
            middle_list = Parallel(n_jobs = -1, verbose = 0)(delayed(get_middle)(i*step) 
                                                                for i in tqdm(range(1,len(self.factor)//step)))   
            tail_dict = get_tail((len(self.factor)//step)* step)

        middle_list.append(head_dict)
        middle_list.append(tail_dict)
        result = middle_list


        date_list = []
        ic_list = []
        exchange_list = []
        long_ret = []
        short_ret = []
        for i in result:
            str_date = i['date'].strftime("%Y-%m-%d")
            date_list.append(datetime.strptime(str_date,"%Y-%m-%d"))
            ic_list.append([i['ic'],i['rank_ic']])
            
            group_long = []
            group_short = []
            group_exchange = []
            for g in range(1,n_group+1):
                group_long.append(i['group_'+str(g)]['long'])
                group_short.append(i['group_'+str(g)]['short'])
                group_exchange.append(i['group_'+str(g)]['exchange'])
            long_ret.append(group_long)
            short_ret.append(group_short)
            exchange_list.append(group_exchange)

        df_ic = pd.DataFrame(ic_list,index=date_list,columns=['ic','rank_ic']).sort_index(ascending=True)
        df_exchange = pd.DataFrame(exchange_list,index=date_list,columns=range(1,n_group+1)).sort_index(ascending=True)
        df_long_ret = pd.DataFrame(long_ret,index=date_list,columns=range(1,n_group+1)).sort_index(ascending=True)
        df_short_ret = pd.DataFrame(short_ret,index=date_list,columns=range(1,n_group+1)).sort_index(ascending=True)
        df_long_ret

        tmp_results = {
            'adj_period': step,
            'ic': df_ic,
            'ret': {'long':df_long_ret.T,'short':df_short_ret.T},
            'exchange': df_exchange.T
        }


        return tmp_results

    def calc_multiple_periods(self):
        results = []
        for i in self.list_periods:
            print("\n{:-^120s}".format('计算调仓周期 = '+str(i)))
            results.append(self.calc_results(i))

        self.results = results
        return results

    # ==============================
    # ==============================
    # IC的分析
    # ==============================
    # ==============================
    # 计算IC和分组IC的样本统计量
    def summary_ic_table(self, start=False, end=False, kind='rank_ic'):
        """
        对于ic的综合描述，包括均值，ir，显著性，偏度，峰度
        :param start: 开始日期，如’2021-01-01‘
        :param end: 结束日期，如’2022-01-01‘
        :param kind: 计算ic的方式，默认是spearman
        :return: 两个dataframe
        """
        if not start:
            start = self.factor.index[0]
        if not end:
            end = self.factor.index[-1]
        summary_ic = pd.DataFrame()
        for result in self.results:
            adj_period = result['adj_period']
            # ==============================
            # IC的分析
            # ==============================
            ic = result['ic'].copy()
            ic = ic[(ic.index >= start) & (ic.index <= end)]
            # 计算ic_mean
            ic_mean = ic.mean()
            ic_std = ic.std(ddof=1)
            # 计算ic_ir
            ic_ir = np.abs(ic_mean) / ic_std
            # 偏度
            ic_skew = ic.skew()
            # 峰度
            ic_kurtosis = ic.kurtosis()
            # 检验是否显著为0
            t_0 = np.sqrt(len(ic)) * ic_mean / ic_std
            p_0 = np.abs(t_0).apply(lambda x: (1-scipy.stats.t.cdf(x, len(ic)-1)) / 2)
            df_tmp1 = pd.DataFrame(data=[ic_mean, ic_std,ic_ir,
                                 ic_skew, ic_kurtosis,
                                 t_0, p_0],
                                   index=['IC Mean','IC Std', 'IC IR',
                                          'IC Skew', 'IC Kurtosis',
                                          't: IC!=0', 'p: IC!=0'])
            df_tmp1.columns = [[f'period_{adj_period}', f'period_{adj_period}'],
                               ['IC', 'rank IC']]

            df_tmp1 = df_tmp1.round(4)
            summary_ic = pd.concat([summary_ic, df_tmp1], axis=1)

        dict1 = {}
        if kind == 'rank_ic':
            df_ic = summary_ic.iloc[:, list(np.arange(0, 2*len(self.results), 2)+1)]
            df_ic.columns = [f'{x[0]}: {x[1]}' for x in df_ic.columns]
            dict1['ic'] = df_ic
            return dict1
        elif kind == 'ic':
            df_ic = summary_ic.iloc[:, list(np.arange(0, 2*len(self.results), 2))]
            df_ic.columns = [f'{x[0]}: {x[1]}' for x in df_ic.columns]
            dict1['ic'] = df_ic
            return dict1
        
    # 画ic的累计变化曲线图
    def plot_cum_ic(self,  kind='rank_ic'):
        fig = plt.figure(figsize=(10, 5*len(self.results)))
        for i in range(len(self.results)):
            adj_period = self.results[i]['adj_period']
            ic = self.results[i]['ic'][kind].to_frame()
            ic.index = pd.to_datetime(ic.index)
            ic.sort_index(ascending=True, axis=1, inplace=True)
            ic['cum'] = np.cumsum(ic[kind])
        
            ax = fig.add_subplot(100*(len(self.results)) + 10 + (i+1))
            ax.plot(ic.index, ic['cum'], alpha=1, color='C0')
            ax.set_title(f'Adj_periods_{adj_period}: {kind}', fontsize=15)
            ax.legend(['cumulative'], loc='upper left')
        plt.tight_layout()
        plt.show()

    # 画ic的变化柱状图，可以按年或者按月
    def plot_ic_bar(self, frequency='Y', kind='rank_ic'):
        df_ic = pd.DataFrame()
        for result in self.results:
            ic = result['ic'].loc[:, kind].copy()
            ic = ic.resample(frequency, axis=0, label='right',
                             closed='right', kind='period').mean().to_frame()
            ic = (100 * ic).round(2)
            ic.columns = [result['adj_period']]
            df_ic = pd.concat([df_ic, ic], axis=1)
        df_ic.plot(kind='bar', figsize=(10, 5))  # edgecolor='white', linewidth=5
        plt.title('IC (%)', fontsize=15)
        plt.legend(fontsize=15)
        plt.show()

    # 画多空收益率按30天滚动和每日的IC
    # 在legend中，plot的优先级高于bar
    def plot_daily_ic(self, kind='rank_ic'):
        fig = plt.figure(figsize=(10, 5*len(self.results)))
        for i in range(len(self.results)):
            adj_period = self.results[i]['adj_period']
            ic = self.results[i]['ic'][kind].to_frame()
            ic = (100 * ic).round(2)
            ic['30d'] = ic[kind].rolling(window=int(20//adj_period)+1).mean()
            ic.index = pd.to_datetime(ic.index)
            ic.sort_index(ascending=True, axis=1, inplace=True)
            ax = fig.add_subplot(100*(len(self.results)) + 10 + (i+1))
            ax.bar(ic.index, ic[kind], alpha=0.6, width=2)
            ax.plot(ic.index, ic['30d'], alpha=1, color='red')
            ax.set_title(f'Adj_periods_{adj_period}: {kind} (%)', fontsize=15)
            ax.legend(['30d', 'daily'], loc='upper left')  # 先写plot的，再写bar的
            ax.axhline(3,color='black',linestyle='--')
            ax.axhline(-3,color='black',linestyle='--')
            ax.fill_between(ic.index, 3, -3,color='yellow',alpha=0.4)
            ax.set_ylim(-40, 40)
        plt.tight_layout()
        plt.show()


    # plot_monthly_IC
    def plot_monthly_ic(self, kind='rank_ic'):
        plt.style.use('default')
        fig = plt.figure(figsize=(10, 5*len(self.results)))
        for i in range(len(self.results)):
            adj_period = self.results[i]['adj_period']
            ic = self.results[i]['ic'][kind].to_frame()
            ic = ic.resample('M', axis=0, label='right',
                             closed='right', kind='period').mean()
            ic.columns = [kind]
            ic['year'] = ic.index.year
            ic['month'] = ic.index.month
            ic = ic.pivot_table(index=['year'], columns=['month'],
                                values=[kind])
            ic.columns = [x[1] for x in ic.columns]
            ic = (100 * ic).round(2)
            ax = fig.add_subplot(100*len(self.results) + 10 + (i+1))
            ax.imshow(ic.values, cmap="summer")
            # Rotate the tick labels and set their alignment.
            ax.set_xticks(np.arange(len(ic.columns)), labels=ic.columns, fontsize=15)
            ax.set_yticks(np.arange(len(ic.index)), labels=ic.index, fontsize=15)
            # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            #          rotation_mode="anchor")
            for j in range(len(ic.index)):
                for k in range(len(ic.columns)):
                    text = ax.text(k, j,ic.values[j, k],
                                        ha="center", va="center", color="black")
            ax.set_title(f'Adj_period_{adj_period}: Monthly IC (%)')
        plt.tight_layout()
        plt.show()
        plt.style.use('_mpl-gallery')

    # ==============================
    # 收益率分析
    # ==============================
    # plot_monthly
    def plot_monthly_ret(self):
        plt.style.use('default')
        fig = plt.figure(figsize=(10, 5*len(self.results)))
        for i in range(len(self.results)):
            adj_period = self.results[i]['adj_period']
            ret = self.results[i]['ret'].T.copy()
            ret = (ret.iloc[:, -1] - ret.iloc[:, 0]).to_frame()  # 用普通收益率算多空
            ret = np.log(1+ret)  # 转化为对数收益率方便加和
            ret = ret.resample('M', axis=0, label='right',
                               closed='right', kind='period').sum()
            ret = np.exp(ret) - 1  # 计算出了每个月的普通收益率
            ret.columns = ['ret']
            ret['year'] = ret.index.year
            ret['month'] = ret.index.month
            ret = ret.pivot_table(index=['year'], columns=['month'],
                                  values=['ret'])
            ret.columns = [x[1] for x in ret.columns]
            ret = (100 * ret).round(2)
            ax = fig.add_subplot(100*len(self.results) + 10 + (i+1))
            ax.imshow(ret.values, cmap="summer") # YlGn
            # Rotate the tick labels and set their alignment.
            ax.set_xticks(np.arange(len(ret.columns)), labels=ret.columns, fontsize=15)
            ax.set_yticks(np.arange(len(ret.index)), labels=ret.index, fontsize=15)
            # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            #          rotation_mode="anchor")

            for j in range(len(ret.index)):
                for k in range(len(ret.columns)):
                    text = ax.text(k, j,ret.values[j, k],
                                        ha="center", va="center", color="black")
            ax.set_title(f'Adj_period_{adj_period}: Long Short Monthly Return (%)')

        plt.tight_layout()
        plt.show()
        plt.style.use('_mpl-gallery')
    

    # 做分层收益的表格
    def summary_layer_ret(self,show_group,excess=False,long=True):
       
        def prod(list_1):
            return np.cumprod(list_1)[-1]

        kind = 'long' if long else 'short'
        summary_layer_ret_table = pd.DataFrame()
        for i in range(len(self.results)):
            adj_period = self.results[i]['adj_period']
            rf = self.rf
            ret = self.results[i]['ret'][kind].T.copy()
            ret.index = pd.to_datetime(ret.index)
            adj_period = self.results[i]['adj_period']
            exchange = self.results[i]['exchange'].sort_index(ascending=True).T.copy()
            
            if excess:
                if type(self.benchmark) == bool:
                    ret = ret - ret.mean(axis=1).values.reshape(len(ret), 1)
                else:
                    benchmark = self.benchmark + 1
                    benchmark['new_ret'] = benchmark.rolling(window=adj_period ).apply(prod)
                    benchmark = benchmark.loc[ret.index,'new_ret'] - 1
                    benchmark = benchmark.fillna(0)
                    ret = ret - benchmark.values.reshape(len(ret), 1)
            
            for j in show_group:
                j = j - 1
                group = ret.columns[j]
                ret_sub = ret.iloc[:,j]
                exchange_sub = exchange.iloc[:,j]

                AVolatility = np.std(ret_sub)*np.sqrt(252/adj_period)
                WinningRatio = len(ret_sub[ret_sub > 0])/len(ret_sub[ret_sub != 0])
                PnLRatio = np.mean(ret_sub[ret_sub > 0]) / abs(np.mean(ret_sub[ret_sub < 0]))
                if self.ret_kind == 'prod':
                    ret_sub = np.cumprod(1 + ret_sub, axis=0) 
                    AReturnRate = (ret_sub[-1]/ret_sub[0]) ** (1/(len(ret_sub)*adj_period/252)) - 1
                else:
                    AReturnRate = np.mean(ret_sub , axis=0) * 252 / adj_period
                    ret_sub = 1 + np.cumsum(ret_sub, axis=0)
                SharpeRatio = (AReturnRate-rf)/AVolatility

                ret_sub = ret_sub.to_list()
                low_point = np.argmax((np.maximum.accumulate(ret_sub)- ret_sub)/np.maximum.accumulate(ret_sub))
                if low_point == 0:
                    MaxDrawdown = 0
                high_point = np.argmax(ret_sub[:low_point])
                MaxDrawdown = (ret_sub[high_point] - ret_sub[low_point]) / ret_sub[high_point]
                Calmar = AReturnRate / MaxDrawdown
                ExchangeMean = np.mean(exchange_sub)

                df_tmp = pd.DataFrame(data=[AReturnRate, AVolatility,
                                            MaxDrawdown,SharpeRatio,Calmar,
                                            WinningRatio, PnLRatio, ExchangeMean],
                                      index=['AReturnRate', 'AVolatility',
                                              'MaxDrawdown','SharpeRatio','Calmar',
                                             'WinningRatio', 'PnLRatio', 'ExchangeMean'],
                                      columns = [f'period{adj_period } group{group}'])
                summary_layer_ret_table = pd.concat([summary_layer_ret_table, df_tmp], axis=1)
        return summary_layer_ret_table

    # 画每一组的收益率
    def plot_layer_ret_bar(self, excess=True, long=True):
        df_ret = pd.DataFrame()
        
        def prod(list_1):
            return np.cumprod(list_1)[-1]
        kind = 'long' if long else 'short'
        for i in range(len(self.results)):
            adj_period = self.results[i]['adj_period']
            ret = self.results[i]['ret'][kind].T.copy()
            ret.columns = list(range(1, self.n_group+1))
            ret.index = pd.to_datetime(ret.index)
            
            if excess:
                if type(self.benchmark) == bool:
                    ret = ret - ret.mean(axis=1).values.reshape(len(ret), 1)
                else:
                    benchmark = self.benchmark + 1
                    benchmark['new_ret'] = benchmark.rolling(window=adj_period ).apply(prod)
                    benchmark = benchmark.loc[ret.index,'new_ret'] - 1
                    benchmark = benchmark.fillna(0)
                    ret = ret - benchmark.values.reshape(len(ret), 1)
            if self.ret_kind == 'prod':
                ret = np.cumprod(1 + ret , axis=0) - 1
                ret = (1 + ret) ** (1 / (len(ret) * adj_period)) - 1
                # ret.dropna(axis=0, inplace=True)
                ret = 10000 * ret.iloc[-1, :].to_frame()
            else:
                ret = 10000 * np.mean(ret , axis=0).to_frame() / adj_period
            ret.columns = [adj_period]
            df_ret = pd.concat([df_ret, ret], axis=1)
        df_ret.plot(kind='bar', figsize=(10, 5))
        plt.title('Group Return', fontsize=15)
        plt.ylabel('daily return (bps)', fontsize=15)
        plt.legend(fontsize=15)
        plt.show()

    # 得到每一组的收益率表格
    def layer_ret_df(self):
        self.calc_multiple_periods()

        df_ret = pd.DataFrame()
        for i in range(len(self.results)):
            ret = self.results[i]['ret'].T.copy()
            ret.columns = list(range(1, self.n_group+1))
            ret.index = pd.to_datetime(ret.index)
            df_ret = pd.concat([df_ret, ret], axis=1)
        return df_ret
        


    # 画分层收益率的图
    def plot_layer_ret(self, excess=True, long=True):
        def prod(list_1):
            return np.cumprod(list_1)[-1]
        
        cmap = cm.get_cmap("RdYlGn")
        cmap = cmap(np.linspace(0, 1, self.n_group))
        fig = plt.figure(figsize=(10, 5*len(self.results)))

        kind = 'long' if long else 'short'
        for i in range(len(self.results)):
            adj_period = self.results[i]['adj_period']
            ret = self.results[i]['ret'][kind].T.copy()
            ret.index = pd.to_datetime(ret.index)
           
            if excess:
                if type(self.benchmark) == bool:
                    ret = ret - ret.mean(axis=1).values.reshape(len(ret), 1)
                else:
                    benchmark = self.benchmark + 1
                    benchmark['new_ret'] = benchmark.rolling(window=adj_period ).apply(prod)
                    benchmark = benchmark.loc[ret.index,'new_ret'] - 1
                    benchmark = benchmark.fillna(0)
                    ret = ret - benchmark.values.reshape(len(ret), 1)
            # ret = np.cumprod(1 + ret, axis=0)
            ret = np.cumprod(1 + ret, axis=0) if self.ret_kind == 'prod' else 1 + np.cumsum(ret, axis=0)
            ax = fig.add_subplot(100*len(self.results) + 10 + (i+1))
            for j in range(len(ret.columns)):
                ax.plot(ret.index, ret.iloc[:, j].values, color=cmap[j], alpha=1)
            ax.legend(ret.columns, loc='upper left', fontsize=10)
            adj_period = self.results[i]['adj_period']
            ax.set_title(f'Adj_periods_{adj_period}: Net Value for Groups', fontsize=15)
            ax.axhline(1, linestyle='--', c='grey')
        plt.tight_layout()
        plt.show()

    # 画表现最好一组的按30天滚动和每日的换手率
    # 在legend中，plot的优先级高于bar
    def plot_daily_exchange(self):
        fig = plt.figure(figsize=(10, 5*len(self.results)))
        for i in range(len(self.results)):
            adj_period = self.results[i]['adj_period']
            exchange = self.results[i]['exchange'].sort_index(ascending=True).iloc[-1,:].T.to_frame('daily')
            exchange = (100 * exchange).round(2)
            smooth = int(20//adj_period)+1
            exchange['30d'] = exchange.rolling(window=smooth).mean()
            exchange.index = pd.to_datetime(exchange.index)
            exchange.sort_index(ascending=True, axis=1, inplace=True)
            ax = fig.add_subplot(100*(len(self.results)) + 10 + (i+1))
            mean_ex = round(np.mean(exchange['daily']),2)
            ax.axhline(mean_ex,color='k',linestyle='--')
            ax.bar(exchange.index, exchange['daily'], alpha=0.6, width=2)
            ax.plot(exchange.index, exchange['30d'], alpha=1, color='red')
            ax.set_title(f'Adj_periods_{adj_period}: exchange_rate (%)', fontsize=15)
            ax.legend([f'mean:{mean_ex}','30d', 'daily'], loc='upper left')  # 先写plot的，再写bar的
            up = min(mean_ex + 2 * np.std(exchange['daily']),100)
            down = mean_ex - 2 * np.std(exchange['daily'])
            # ax.fill_between(exchange.index, np.mean(exchange['daily']), 0,color='yellow',alpha=0.4)
            ax.set_xlim(exchange.index[0], exchange.index[-1])
            ax.set_ylim(down, up)
        plt.tight_layout()
        plt.show()


    # ==============================
    # ==============================
    # 快速分析
    # ==============================
    # ==============================
    def fast_analysis(self):
        self.calc_multiple_periods()
        
        print("\n{:-^120s}".format('Long Returns Performance'))
        self.plot_layer_ret_bar(excess=False, long=True)
        summary_layer_ret = self.summary_layer_ret(show_group=[1,self.n_group],excess=False,long=True)
        print(summary_layer_ret)
        self.plot_layer_ret(excess=False, long=True)
        
        print("\n{:-^120s}".format('Short Returns Performance'))
        self.plot_layer_ret_bar(excess=False,long=False)
        summary_layer_ret = self.summary_layer_ret(show_group=[1,self.n_group],excess=False,long=False)
        print(summary_layer_ret)
        self.plot_layer_ret(excess=False, long=False)

        print("\n{:-^120s}".format('Excess Long Returns Performance'))
        self.plot_layer_ret_bar(excess=True, long=True)
        summary_layer_ret = self.summary_layer_ret(show_group=[1,self.n_group],excess=True,long=True)
        print(summary_layer_ret)
        self.plot_layer_ret(excess=True, long=True)

        print("\n{:-^120s}".format('Exchange_rate of Best Group'))
        self.plot_daily_exchange()

        # ic的部分
        print("{:=^120s}".format('IC'))
        summary_ic_table = self.summary_ic_table()
        print("\n{:-^120s}".format('IC performance'))
        print(summary_ic_table['ic'])

        print("\n{:-^120s}".format('IC yearly'))
        self.plot_ic_bar('Y', kind='rank_ic')
        # print("\n{:-^120s}".format('IC monthly'))
        # self.plot_monthly_ic(kind='rank_ic')
        print("\n{:-^120s}".format('IC daily'))
        self.plot_daily_ic(kind='rank_ic')
        print("\n{:-^120s}".format('Cumulative IC'))
        self.plot_cum_ic(kind='rank_ic')

    def hedge_preformance(self,hedge_group):
        fig = plt.figure(figsize=(10, 5*len(self.results)))
        summary_layer_ret_table = pd.DataFrame()
        for i in range(len(self.results)):
            adj_period = self.results[i]['adj_period']
            rf = self.rf
            ret_long = self.results[i]['ret']['long'].T.copy().iloc[:,hedge_group[1]-1].to_frame()
            group_long = ret_long.columns[0]
            ret_short = self.results[i]['ret']['short'].T.copy().iloc[:,hedge_group[0]-1].to_frame()
            group_short = ret_short.columns[0]
            ret_hedge = (ret_long.iloc[:,0] + ret_short.iloc[:,0])/2
            ret_hedge = ret_hedge.to_frame('hedge')

            # 作图
            if self.ret_kind == 'prod': 
                values_df = np.cumprod(1 + ret_hedge)
                values_df[group_long] = np.cumprod(1 + ret_long)
                values_df[group_short] = np.cumprod(1 - ret_short)
            else:
                values_df = 1 + np.cumsum(ret_hedge, axis=0)
                values_df[group_long] = 1 + np.cumsum(ret_long, axis=0)
                values_df[group_short] = 1 - np.cumsum(ret_short, axis=0)
            ax = fig.add_subplot(100*len(self.results) + 10 + (i+1))
            for j in range(len(values_df.columns)):
                ax.plot(values_df.index, values_df.iloc[:, j].values, alpha=1)
            ax.legend(values_df.columns, loc='upper left', fontsize=10)
            ax.set_title(f'Adj_periods_{adj_period}: Net Value for Groups', fontsize=15)
            ax.axhline(1, linestyle='--', c='grey')

            adj_period = self.results[i]['adj_period']
            exchange_long = self.results[i]['exchange'].sort_index(ascending=True).T.copy().iloc[:,hedge_group[1]-1]
            exchange_short = self.results[i]['exchange'].sort_index(ascending=True).T.copy().iloc[:,hedge_group[0]-1]
            exchange_hedge = (exchange_long + exchange_short) / 2
            show_group = [[group_long,ret_long,exchange_long],
                          [group_short,ret_short,exchange_short],
                          ['hedege',ret_hedge,exchange_hedge]]
            for n in show_group:
                group = n[0]
                ret_sub = n[1].iloc[:,0]
                exchange_sub = n[2]
                # ret_hedge.index = pd.to_datetime(ret.index)

                AVolatility = np.std(ret_sub)*np.sqrt(252/adj_period)
                WinningRatio = len(ret_sub[ret_sub > 0])/len(ret_sub[ret_sub != 0])
                PnLRatio = np.mean(ret_sub[ret_sub > 0]) / abs(np.mean(ret_sub[ret_sub < 0]))
                if self.ret_kind == 'prod':
                    ret_sub = np.cumprod(1 + ret_sub, axis=0) 
                    AReturnRate = (ret_sub[-1]/ret_sub[0]) ** (1/(len(ret_sub)*adj_period/252)) - 1
                else:
                    AReturnRate = np.mean(ret_sub , axis=0) * 252 / adj_period
                    ret_sub = 1 + np.cumsum(ret_sub, axis=0)
                SharpeRatio = (AReturnRate-rf)/AVolatility

                ret_sub = ret_sub.to_list()
                low_point = np.argmax((np.maximum.accumulate(ret_sub)- ret_sub)/np.maximum.accumulate(ret_sub))
                if low_point == 0:
                    MaxDrawdown = 0
                high_point = np.argmax(ret_sub[:low_point])
                MaxDrawdown = (ret_sub[high_point] - ret_sub[low_point]) / ret_sub[high_point]
                Calmar = AReturnRate / MaxDrawdown
                ExchangeMean = np.mean(exchange_sub)

                df_tmp = pd.DataFrame(data=[AReturnRate, AVolatility,
                                            MaxDrawdown,SharpeRatio,Calmar,
                                            WinningRatio, PnLRatio, ExchangeMean],
                                      index=['AReturnRate', 'AVolatility',
                                              'MaxDrawdown','SharpeRatio','Calmar',
                                             'WinningRatio', 'PnLRatio', 'ExchangeMean'],
                                      columns = [f'period{adj_period } group{group}'])
                summary_layer_ret_table = pd.concat([summary_layer_ret_table, df_tmp], axis=1)
        print(summary_layer_ret_table)
        plt.tight_layout()
        plt.show()
        return summary_layer_ret_table
        

        
        
            
    






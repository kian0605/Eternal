#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 12:24:39 2022

@author: k&J
2023/2/1
Fix the issue of producing different forecast results
1. Modify the Impyute function to ensure the imputation has the same random seed. Then the dataset_X function can consistently have the same results
2. Impose the random state in each estimation function: BR, GBA and ABR
3. Remove the calculation of dataset_X function in fcast function

2023/2/20
1. Combine the code from Janice (Eternal_project_pickle.py)

"""
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta as rd
from datetime import datetime
#---------------------------
import matplotlib.pyplot as plt
import warnings

from numpy.linalg import svd
import impyute as impy
from statsmodels.tsa.stattools import adfuller as dftest

import seaborn as sns; sns.set()

from sklearn.linear_model import LassoCV
from sklearn.ensemble import BaggingRegressor as BR
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import AdaBoostRegressor as ABR
from sklearn.metrics import mean_squared_error as mse

from sklearn.model_selection import GridSearchCV

from impute import em1

warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 

###------------- 中文字型設定
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
#plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100



def my_date_parser(dt):
    return datetime(np.int(dt[0:4]), np.int(dt[5:7]),1)

def nrmlize(x): 
    xmean = np.mean(x,0) #算平均，逗號後的 0 是表達" axis = 0：壓縮行，對各列求均值，返回 1 * n 矩陣 (是最常用的，相當於對各欄變數取值)"
    xmad=np.mean((x-xmean)**2,0) #算變異數
    z = (x-xmean)/np.sqrt(xmad) #標準化(分母是標準差)
    return z #設定此 def 最終要回傳數值

def remove_outliers(X,c): 
    """
    Function that removes outliers from the data. 
    A data point x is considered an outlier if |x-median|>10*interquartile_range. 

    Parameters
    ----------    
    X : dataset 
    
    Returns
    -------
    Y : dataset with outliers replaced with NaN
    n : number of outliers removed from each series
    """
    # np.tile 是將原來資料中每一維度的元素進行 copy
    median_X_mat = np.tile(X.quantile(0.5),(np.size(X,0),1))    
    IQR_mat = np.tile(X.quantile(0.75)-X.quantile(0.25),(np.size(X,0),1))   
    Z = abs(X-median_X_mat)
    outlier = Z>(c*IQR_mat)
    n = outlier.sum()
    # 若對應的 outlier 為 False，會保留原始值；為 True，則以 nan 替換。
    Y = X.mask(outlier) 
    return [Y,n]



class Eternal_project:
    """
    Eternal_project 是
    
    參數設定
   ----------
       
   方法(method)
   ------------
   dataset_X(dataset): 執行此函數必須給定參數 dataset (代表資料 X 的資料集為原物料或是CEIC等資料庫) 
   此函數用來針對 X 資料進行差補轉換等
   並用此處理過後之資料進行 PCA 轉換成主成份
   會產生 attribute : self.X 以及 self.X
   """
    
    def __init__(self,y,X,path,h0,grid,scenario): # Janice 抽調 tt0,tt1,
        from datetime import datetime
        self.y1 = y
        self.path = path
        self.h0 = h0
        self.grid = grid
        self.data = X
        self.cat_list = list(self.data.keys())
        #self.tt0 = tt0
        #self.tt1 = tt1
        self.today = datetime.today().strftime('%Y%m') #計有估計結果是使用today日期作為資料夾(舉例202301,及代表2023年1月)
        # self.today = '202301'  # 暫時綁定為固定資料夾名稱
        self.scenario = scenario
        if scenario == 'covid': # Janice 拉過來的
            self.tt0 = pd.Timestamp('2020-1-1', freq='MS')
            self.tt1 = pd.Timestamp('2021-12-1', freq='MS')
        elif scenario == 'normal':
            self.tt0 = pd.Timestamp('2017-1-1', freq='MS')
            self.tt1 = pd.Timestamp('2020-12-1', freq='MS')

        
    def dataset_X(self,dataset): 
        # All X
        self.dataset = dataset
        if self.dataset == 'all':
            t0 = self.data[self.cat_list[-1]].dropna(how='any',axis=0).index[0]
            t1 = self.data[self.cat_list[-1]].dropna(how='any',axis=0).index[-1]
            X = pd.concat(self.data, axis=1, ignore_index=False)
            X = X.replace({0:np.nan})
            X ,n = remove_outliers(X[t0:t1],10)
            Xrolling = X.rolling(12, min_periods=1, center=False).mean()
            X.update(Xrolling, overwrite=False)
            X = X.interpolate(limit_area='inside')
            if np.sum(np.sum(X.isna()))>0:
                X = pd.DataFrame(em1(np.array(X).T, loops=50).T,index = X.index,columns=X.columns)
            X = X.dropna(how='any',axis=0) 
            for jj in range(X.shape[1]):
                dfresult = dftest(X.iloc[:,jj],regression='c')
                if dfresult[1]>0.1:
                    X.iloc[:,jj] = X.iloc[:,jj].pct_change(12)
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.dropna(how='any',axis=0) 
            X = nrmlize(X)
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.dropna(how='any',axis=1) 
            X.columns.names = ['CEIC分類','變數名稱']
            self.X = X
        elif self.dataset == '20':
            t0 = self.data[self.cat_list[-1]].dropna(how='any',axis=0).index[0]
            t1 = self.data[self.cat_list[-1]].dropna(how='any',axis=0).index[-1]
            X = pd.concat(self.data, axis=1, ignore_index=False)
            X ,n = remove_outliers(X[t0:t1],10)
            X = X.replace({0:np.nan})
            Xrolling = X.rolling(12, min_periods=1, center=False).mean()
            X.update(Xrolling, overwrite=False)
            X = X.interpolate(limit_area='inside')
            if np.sum(np.sum(X.isna()))>0:
                X = pd.DataFrame(em1(np.array(X).T, loops=50).T,index = X.index,columns=X.columns)
            X = X.dropna(how='any',axis=0) 
            for jj in range(X.shape[1]):
                dfresult = dftest(X.iloc[:,jj],regression='c')
                if dfresult[1]>0.1:
                    X.iloc[:,jj] = X.iloc[:,jj].pct_change(12)
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.dropna(how='any',axis=0) 
            X = nrmlize(X)
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.dropna(how='any',axis=1) 
            X.columns.names = ['CEIC分類','變數名稱']
            X = X[self.cat_list[-1:]]
            self.X = X
        '''   
        elif self.dataset == 'long':
            t0 = self.data[self.cat_list[0]].index[0]
            t1 = self.data[self.cat_list[0]].index[-1]
            X = pd.concat(self.data, axis=1, ignore_index=False)
            X = X[self.cat_list[:-1]]
            X ,n = remove_outliers(X[t0:t1],10)
            X = X.replace({0:np.nan})
            Xrolling = X.rolling(12, min_periods=1, center=False).mean()
            X.update(Xrolling, overwrite=False)
            X = X.interpolate(limit_area='inside')
            if np.sum(np.sum(X.isna()))>0:
                X = pd.DataFrame(impy.imputation.cs.em(np.array(X).T, loops=50).T,index = X.index,columns=X.columns)
            X = X.dropna(how='any',axis=0) 
            for jj in range(X.shape[1]):
                dfresult = dftest(X.iloc[:,jj],regression='c')
                if dfresult[1]>0.1:
                    X.iloc[:,jj] = X.iloc[:,jj].pct_change(12)
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.dropna(how='any',axis=0) 
            X = nrmlize(X)
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.dropna(how='any',axis=1) 
            X.columns.names = ['CEIC分類','變數名稱']
            self.X = X
            '''
        u,s,v = svd(self.X,full_matrices=False)
        r = min(25,self.X.shape[1])
        f = np.sqrt(len(u))*u[:,:r] 
        self.f = pd.DataFrame(f,index=self.X.index) 
        loading = v[:np.min(self.X.shape),:].T@np.linalg.pinv(np.diag(s))
        self.f_loading = loading[:,:r]
        self.x_loading = pd.DataFrame(np.zeros((len(self.h0),self.X.shape[1])),columns=self.X.columns)
        return [self.X,self.f]

    def est(self,name): #解開來 只跑單一y
        import os
        import pickle
        from sklearn.model_selection import ShuffleSplit
        from sklearn.model_selection import RepeatedKFold
        #cv = RepeatedKFold(n_splits=2, n_repeats=20, random_state=1)
        # renew time index, t0 and t1
        #tt0 = X.index[0]
        self.name = name
        tt2 = self.X.index[-1]
        self.estimators = {}
        for ii,h in enumerate(self.h0):
            self.estimators['h='+str(h)] = {}

            y_0 = pd.DataFrame(self.y1[name].dropna())
            t0 = max(self.tt0,y_0.index[0])
            # out-of-sample index
            t2 = min(tt2,y_0.index[-1])
            # in-sample index
            t1b = t2-rd(months=h)
            t1a = self.tt1

            y_h = y_0[t0+rd(months=h):t1a]
            # Traditional correlation
            X_h = self.X[t0:t1a-rd(months=h)]
# =============================================================================
#             z = pd.concat([y_h,X_h],axis=1)
#             z = z.loc[:,z.var()!=0]
#             z_corr = z.corr().iloc[0,:]
#             z_sort = z_corr.sort_values().dropna()
#             x_negcorr = z_sort[:10]
#             x_postcorr= z_sort[-11:-1].sort_values(ascending=False)
# =============================================================================


            f_h = self.f[t0:t1a-rd(months=h)]


            # BR, GBR, ABR (grid search)
            reg = LassoCV(cv=RepeatedKFold(n_splits=2, n_repeats=20, random_state=1), random_state=1, max_iter=10000).fit(f_h, y_h.iloc[:,0])
            grid_models = GridSearchCV(estimator=BR(random_state=1), param_grid={key: self.grid[key] for key in ['n_estimators','max_samples']}, n_jobs=-1, cv=RepeatedKFold(n_splits=2, n_repeats=20, random_state=1)).fit(f_h, y_h.iloc[:,0])
            reg_BR = grid_models.best_estimator_
            grid_models = GridSearchCV(estimator=GBR(random_state=1), param_grid={key: self.grid[key] for key in ['n_estimators','learning_rate','subsample','max_depth']}, n_jobs=-1, cv=RepeatedKFold(n_splits=2, n_repeats=20, random_state=1)).fit(f_h, y_h.iloc[:,0])
            reg_GBR = grid_models.best_estimator_
            grid_models = GridSearchCV(estimator=ABR(random_state=1), param_grid={key: self.grid[key] for key in ['n_estimators','learning_rate']}, n_jobs=-1, cv=RepeatedKFold(n_splits=2, n_repeats=20, random_state=1)).fit(f_h, y_h.iloc[:,0])
            reg_ABR = grid_models.best_estimator_

            # BRx, GBRx, ABRx (grid search)
            regx = LassoCV(cv=RepeatedKFold(n_splits=2, n_repeats=20, random_state=1), random_state=1, max_iter=10000).fit(X_h, y_h.iloc[:,0])
            grid_models = GridSearchCV(estimator=BR(random_state=1), param_grid={key: self.grid[key] for key in ['n_estimators','max_samples']}, n_jobs=-1, cv=ShuffleSplit(n_splits=5, test_size=.25, random_state=1)).fit(X_h, y_h.iloc[:,0])
            reg_BRx = grid_models.best_estimator_
            grid_models = GridSearchCV(estimator=GBR(random_state=1), param_grid={key: self.grid[key] for key in ['n_estimators','learning_rate','subsample','max_depth']}, n_jobs=-1, cv=ShuffleSplit(n_splits=5, test_size=.25, random_state=1)).fit(X_h, y_h.iloc[:,0])
            reg_GBRx = grid_models.best_estimator_
            grid_models = GridSearchCV(estimator=ABR(random_state=1), param_grid={key: self.grid[key] for key in ['n_estimators','learning_rate']}, n_jobs=-1, cv=ShuffleSplit(n_splits=5, test_size=.25, random_state=1)).fit(X_h, y_h.iloc[:,0])
            reg_ABRx = grid_models.best_estimator_

            self.estimators['h='+str(h)].update({name:[reg,reg_BR,reg_GBR,reg_ABR,regx,reg_BRx,reg_GBRx,reg_ABRx]})
            if os.path.exists(os.path.join(self.path, self.today, self.dataset, self.scenario)) == False:
                os.makedirs(os.path.join(self.path, self.today, self.dataset, self.scenario))
            self.save_path = os.path.join(self.path, self.today, self.dataset, self.scenario)
            with open(self.save_path + '/'+name+'_estimator.pickle', 'wb') as f:
                pickle.dump(self.estimators, f)
        return self.estimators

    
    def importances(self,h,name): #  name 表示選單一 y
        import pickle
        import tkinter as tk
        from tkinter import filedialog
        self.name = name
        n1 = self.f.shape[1]
        n2 = self.X.shape[1]
        self.feature_f_importance = np.zeros((n1,4))
        self.feature_x_importance = np.zeros((n2,4))
        if hasattr(self,'estimators')==True:
            self.feature_f_importance[:,0] = self.estimators['h='+str(h)][name][0].coef_
            reg_BR = self.estimators['h='+str(h)][name][1]
            importance = np.zeros((n1,len(reg_BR.estimators_)))
            for ii in range(len(reg_BR.estimators_)):
                importance[:,ii] = reg_BR.estimators_[ii].feature_importances_
            self.feature_f_importance[:,1] = np.mean(importance,axis=1)
            self.feature_f_importance[:,2] = self.estimators['h='+str(h)][name][2].feature_importances_
            self.feature_f_importance[:,3] = self.estimators['h='+str(h)][name][3].feature_importances_
            
            self.feature_x_importance[:,0] = self.estimators['h='+str(h)][name][4].coef_
            reg_BRx = self.estimators['h='+str(h)][name][5]
            importance = np.zeros((n2,len(reg_BRx.estimators_)))
            for ii in range(len(reg_BRx.estimators_)):
                importance[:,ii] = reg_BRx.estimators_[ii].feature_importances_
            self.feature_x_importance[:,1] = np.mean(importance,axis=1)
            self.feature_x_importance[:,2] = self.estimators['h='+str(h)][name][6].feature_importances_
            self.feature_x_importance[:,3] = self.estimators['h='+str(h)][name][7].feature_importances_
        else:
            '''
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename()
            '''
            import os # Janice 加入
            file_path = os.path.join(self.path, self.today, self.dataset, self.scenario) # Janice 修改過的地方
            with open(file_path + '/'+name+'_estimator.pickle', 'rb') as f:
                estimators = pickle.load(f)
            self.feature_f_importance[:,0] = estimators['h='+str(h)][name][0].coef_
            reg_BR = estimators['h='+str(h)][name][1]
            importance = np.zeros((n1,len(reg_BR.estimators_)))
            for ii in range(len(reg_BR.estimators_)):
                importance[:,ii] = reg_BR.estimators_[ii].feature_importances_
            self.feature_f_importance[:,1] = np.mean(importance,axis=1)
            self.feature_f_importance[:,2] = estimators['h='+str(h)][name][2].feature_importances_
            self.feature_f_importance[:,3] = estimators['h='+str(h)][name][3].feature_importances_
            
            self.feature_x_importance[:,0] = estimators['h='+str(h)][name][4].coef_
            reg_BRx = estimators['h='+str(h)][name][5]
            importance = np.zeros((n2,len(reg_BRx.estimators_)))
            for ii in range(len(reg_BRx.estimators_)):
                importance[:,ii] = reg_BRx.estimators_[ii].feature_importances_
            self.feature_x_importance[:,1] = np.mean(importance,axis=1)
            self.feature_x_importance[:,2] = estimators['h='+str(h)][name][6].feature_importances_
            self.feature_x_importance[:,3] = estimators['h='+str(h)][name][7].feature_importances_
            

    def fcast(self,name): #  name 表示選單一 y
        import pickle
        #import tkinter as tk
        #from tkinter import filedialog
        self.name = name

        # renew time index, t0 and t1
        #tt0 = X.index[0]
        tt2 = self.X.index[-1]
        # the following code is removed on 2023/2/1 because self.f is calculated from using dataset_X
        #u,s,v = svd(self.X,full_matrices=False)
        #r = min(25,self.X.shape[1])

        #f = np.sqrt(len(u))*u[:,:r]
        #self.f = pd.DataFrame(f,index=self.X.index)
        #loading = v[:np.min(self.X.shape),:].T@np.linalg.pinv(np.diag(s))
        #f_loading = loading[:,:r]
        #x_loading = pd.DataFrame(np.zeros((len(self.h0),self.X.shape[1])),columns=self.X.columns)
        self.result = {}
        if hasattr(self,'estimators')==True:
            pass
        else:
            '''
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename()
            '''
            import os # Janice 加入
            file_path = os.path.join(self.path, self.today, self.dataset, self.scenario) # Janice 修改過的地方
            with open(file_path + '/'+name+'_estimator.pickle', 'rb') as f:
                self.estimators = pickle.load(f)
        for ii, h in enumerate(self.h0):
            self.result['h=' + str(h)] = {}
            #for kk in range(7):  # 把 y index  的kk迴圈拆掉 下方只跑單一被選定的y
            y_0 = pd.DataFrame(self.y1[name].dropna())

            t0 = max(self.tt0, y_0.index[0])
            # out-of-sample index
            t2 = min(tt2, y_0.index[-1])
            #t1b = t2 - rd(months=h)
            # in-sample index
            t1a = self.tt1
            #print(t0,t1a,t1b,tt2)
            y_h = y_0[t0 + rd(months=h):t1a]
            # Traditional correlation
            X_h = self.X[t0:t1a - rd(months=h)]
            z = pd.concat([y_h, X_h], axis=1)
            z = z.loc[:, z.var() != 0]
            z_corr = z.corr().iloc[0, :]
            z_sort = z_corr.sort_values().dropna()
            c = min(X_h.shape[1],10)
            x_negcorr = z_sort[:c]
            x_postcorr = z_sort[-1*c-1:-1].sort_values(ascending=False)
            # X[x_negcorr.index].plot()
            # X[x_postcorr.index].plot()
            f_h = self.f[t0:t1a - rd(months=h)]

            reg = self.estimators['h=' + str(h)][name][0]
            reg_BR = self.estimators['h=' + str(h)][name][1]
            reg_GBR = self.estimators['h=' + str(h)][name][2]
            reg_ABR = self.estimators['h=' + str(h)][name][3]
            regx = self.estimators['h=' + str(h)][name][4]
            reg_BRx = self.estimators['h=' + str(h)][name][5]
            reg_GBRx = self.estimators['h=' + str(h)][name][6]
            reg_ABRx = self.estimators['h=' + str(h)][name][7]

            # in-sample forecast
            yhat_LPCA = reg.predict(f_h)
            yhat_LPCA = pd.DataFrame(yhat_LPCA, index=pd.date_range(t0, t1a - rd(months=h), freq='MS'))

            yhat_Lx = regx.predict(X_h)
            yhat_Lx = pd.DataFrame(yhat_Lx, index=pd.date_range(t0, t1a - rd(months=h), freq='MS'))

            ## bagging pca, boosting pca, adboosting pca
            yhat_BR = reg_BR.predict(f_h)
            yhat_BR = pd.DataFrame(yhat_BR, index=pd.date_range(t0, t1a - rd(months=h), freq='MS'))
            yhat_GBR = reg_GBR.predict(f_h)
            yhat_GBR = pd.DataFrame(yhat_GBR, index=pd.date_range(t0, t1a - rd(months=h), freq='MS'))
            yhat_ABR = reg_ABR.predict(f_h)
            yhat_ABR = pd.DataFrame(yhat_ABR, index=pd.date_range(t0, t1a - rd(months=h), freq='MS'))

            ## bagging X, boosting X, adboosting X
            yhat_BRx = reg_BRx.predict(X_h)
            yhat_BRx = pd.DataFrame(yhat_BRx, index=pd.date_range(t0, t1a - rd(months=h), freq='MS'))
            yhat_GBRx = reg_GBRx.predict(X_h)
            yhat_GBRx = pd.DataFrame(yhat_GBRx, index=pd.date_range(t0, t1a - rd(months=h), freq='MS'))
            yhat_ABRx = reg_ABRx.predict(X_h)
            yhat_ABRx = pd.DataFrame(yhat_ABRx, index=pd.date_range(t0, t1a - rd(months=h), freq='MS'))

            fcast_index = pd.concat(
                [yhat_LPCA, yhat_BR, yhat_GBR, yhat_ABR, yhat_Lx, yhat_BRx, yhat_GBRx, yhat_ABRx, y_h], axis=1)
            self.x_loading.iloc[ii, :] = self.f_loading @ reg.coef_ * np.sqrt(len(self.f))
            x_sort = self.x_loading.iloc[ii, :].sort_values()
            x_negcoef = x_sort[:c]
            x_postcoef = x_sort[-1*c:].sort_values(ascending=False)
            # out-of-sample forecast
            yfcast_LPCA = reg.predict(self.f[t0 + rd(months=h):t2])
            yfcast_LPCA = pd.DataFrame(yfcast_LPCA, index=pd.date_range(t0 + rd(months=2 * h), t2+rd(months=h), freq='MS'))

            yfcast_Lx = regx.predict(self.X[t0 + rd(months=h):t2])
            yfcast_Lx = pd.DataFrame(yfcast_Lx, index=pd.date_range(t0 + rd(months=2 * h), t2+rd(months=h), freq='MS'))

            ## bagging pca, boosting pca, adboosting pca
            yfcast_BR = reg_BR.predict(self.f[t0 + rd(months=h):t2])
            yfcast_BR = pd.DataFrame(yfcast_BR, index=pd.date_range(t0 + rd(months=2 * h), t2+rd(months=h), freq='MS'))
            yfcast_GBR = reg_GBR.predict(self.f[t0 + rd(months=h):t2])
            yfcast_GBR = pd.DataFrame(yfcast_GBR, index=pd.date_range(t0 + rd(months=2 * h), t2+rd(months=h), freq='MS'))
            yfcast_ABR = reg_ABR.predict(self.f[t0 + rd(months=h):t2])
            yfcast_ABR = pd.DataFrame(yfcast_ABR, index=pd.date_range(t0 + rd(months=2 * h), t2+rd(months=h), freq='MS'))

            ## bagging X, boosting X, adboosting X
            yfcast_BRx = reg_BRx.predict(self.X[t0 + rd(months=h):t2])
            yfcast_BRx = pd.DataFrame(yfcast_BRx, index=pd.date_range(t0 + rd(months=2 * h), t2+rd(months=h), freq='MS'))
            yfcast_GBRx = reg_GBRx.predict(self.X[t0 + rd(months=h):t2])
            yfcast_GBRx = pd.DataFrame(yfcast_GBRx, index=pd.date_range(t0 + rd(months=2 * h), t2+rd(months=h), freq='MS'))
            yfcast_ABRx = reg_ABRx.predict(self.X[t0 + rd(months=h):t2])
            yfcast_ABRx = pd.DataFrame(yfcast_ABRx, index=pd.date_range(t0 + rd(months=2 * h), t2+rd(months=h), freq='MS'))

            yfcast = pd.concat(
                [yfcast_LPCA, yfcast_BR, yfcast_GBR, yfcast_Lx, yfcast_ABR, yfcast_BRx, yfcast_GBRx, yfcast_ABRx,
                 y_0[t0 + rd(months=2 * h):t2]], axis=1)
            yfcast.columns = ['forecast_LPC','forecast_BR','forecast_GBR','forecast_ABR','forecast_LX','forecast_BRx','forecast_GBRx','forecast_ABRx','real']
            yfcast['upper'] = yfcast.loc[:,'forecast_LPC':'forecast_ABRx'].max(axis=1)
            yfcast['lower'] = yfcast.loc[:,'forecast_LPC':'forecast_ABRx'].min(axis=1)
            yfcast['mean'] = yfcast.loc[:,'forecast_LPC':'forecast_ABRx'].mean(axis=1)
            # X[x_negcoef.index].plot()
            # X[x_postcoef.index].plot()
            est_mse = np.nanmean((yfcast.iloc[:, :]['2022-1-1':] - np.array(yfcast.iloc[:, 8:9]['2022-1-1':])) ** 2,axis=0)
            est_mse = pd.DataFrame(est_mse,index=yfcast.columns)
            self.result['h=' + str(h)].update(
                {name: [fcast_index, yfcast, x_negcoef, x_postcoef, x_negcorr, x_postcorr, est_mse]})
        return self.result
    
    def plot(self,plot_type):
        import os
        from opencc import OpenCC
        from matplotlib.font_manager import FontProperties
        plt.rcParams['font.sans-serif'] = ['HanWangYenLight'] 
        if os.path.exists(os.path.join(self.path, self.today, self.dataset, self.scenario))==False:
            os.makedirs(os.path.join(self.path, self.today, self.dataset, self.scenario))
        self.save_path = os.path.join(self.path, self.today, self.dataset, self.scenario)
        if plot_type=='importance':
            self.feature_x_importance = pd.DataFrame(self.feature_x_importance,columns=['regx','reg_BRx','reg_GBRx','reg_ABRx'],index=self.X.columns.to_flat_index())
            fig, axes = plt.subplots(nrows=4, ncols=1)
            for ii in range(4):
                c = -1*min(10,len(self.feature_x_importance.iloc[:,0]))
                print(len(self.feature_x_importance.iloc[:,0]))
                df = self.feature_x_importance.iloc[:,ii:ii+1].sort_values(by=[self.feature_x_importance.columns[ii]],ascending=True).iloc[c:,:]
                df.index = list(map(lambda x: OpenCC('s2tw').convert(x[0]+':'+x[1]), df.index))
                df.plot(kind='barh',ax=axes[ii],fontsize=8,figsize=(12,24))
            #plt.subplots_adjust(top=.9,bottom=0.1,left=0.23,right=0.95) #調整圖像位置 
            plt.tight_layout()
            plt.savefig(self.save_path+'/'+self.name+'_importance.png',dpi=300,bbox_inches='tight')
            plt.show() #show一定要用在最後，因為它會將畫布刷新
            plt.close() #關閉圖形視窗
        elif plot_type=='fcast':
            fig, axes = plt.subplots(dpi=100)
            for h in self.h0:
                styles = ['--','--','--','--','--','--','--','--','-']
                linewidths = [1.5, 1.5, 1.5,1.5,1.5,1.5,1.5,1.5,3]
                df = pd.DataFrame(np.array(self.result['h='+str(h)][self.name][1].loc['2017-7-1':,'forecast_LPC':'real']),columns=['forecast_LPC','forecast_BR','forecast_GBR','forecast_ABR','forecast_LX','forecast_BRx','forecast_GBRx','forecast_ABRx','real'],index=self.result['h='+str(h)][self.name][1]['2017-7-1':].index)           
                for col, style, lw in zip(df.columns, styles, linewidths):
                    if col == 'real':
                        df[col].plot(style=style, lw=lw)
                    else:
                        df[col].plot(style=style, lw=lw, alpha=0.7)
                self.result['h='+str(h)][self.name][1]['upper'][self.tt1+rd(months=1):].plot(style='-', lw=2 ,color='grey')
                self.result['h='+str(h)][self.name][1]['lower'][self.tt1+rd(months=1):].plot(style='-', lw=2 ,color='grey')
                self.result['h='+str(h)][self.name][1]['mean'][self.tt1+rd(months=1):].plot(style='-', lw=2 ,color='grey')
                axes.set_title(self.name+' , h='+str(h), x=0.5, y=0.93, fontsize=12) #添加圖像標題，並設定其坐標、字體大小
                axes.legend(['forecast_LPC','forecast_BR','forecast_GBR','forecast_ABR','forecast_LX','forecast_BRx','forecast_GBRx','forecast_ABRx','real','upper','lower','mean'],fontsize=6) #圖例內容、facecolor='w'背景顏色、字體大小    
                #axes[i1,i2].axvline(df.index[-1]-rd(months=h),color='b')
                axes.axvline(self.tt1+rd(months=1),color='b')
                #plt.savefig(path+'/graph/'+est_date+'/'+dataset+'/'+labels+' , h='+str(h)+'_'+ass+'_fcast_index.png',dpi=300)
            plt.savefig(self.save_path+'/'+self.name+'_fcast.png',dpi=300,bbox_inches='tight')
            plt.show() #show一定要用在最後，因為它會將畫布刷新
            plt.close() #關閉圖形視窗
            
    def to_excel(self,name):
        import os
        if os.path.exists(os.path.join(self.path, self.today, self.dataset, self.scenario))==False:
            os.makedirs(os.path.join(self.path, self.today, self.dataset, self.scenario))
        self.save_path = os.path.join(self.path, self.today, self.dataset, self.scenario)
        self.mse = pd.DataFrame(np.zeros((12,1)),index = ['forecast_LPC','forecast_BR','forecast_GBR','forecast_ABR','forecast_LX','forecast_BRx','forecast_GBRx','forecast_ABRx','real','upper','lower','mean'])
        for h in self.h0:
            writer = pd.ExcelWriter(self.save_path+'/'+name+'_'+str(h)+'.xlsx', engine='openpyxl') # 設定路徑及檔名，並指定引擎openpyxl
            self.result['h='+str(h)][name][0].to_excel(writer, sheet_name='(h='+str(h)+')'+' fcast_in') #合成指標與真實指標
            self.result['h='+str(h)][name][1].to_excel(writer, sheet_name='(h='+str(h)+')'+' fcast_out') #合成指標與真實指標
            df_for_x = pd.DataFrame() # 建立一個df，放 x_coef_corr
            df_for_x_data = pd.DataFrame() # 建立一個df，放 
    
            for i,name2 in enumerate(['x_negcoef','x_postcoef','x_negcorr','x_postcorr']): 
                for_x = pd.DataFrame(self.result['h='+str(h)][name][i+2])
                for_x.columns = [name2]                 
                for_x = for_x.reset_index()
                if i+1>2 : #for_x['index'][0].type #是tuple
                    #在位置 0 處建立一個名稱為 'CEIC分類' 的新列，預設值為 nan。
                    #allow_duplicates=False 確保 dataFrame 中只有一列名為 column 的列
                    for_x.insert(0, 'CEIC分類', np.nan, allow_duplicates=False)
                    for_x.insert(1, '變數名稱', np.nan, allow_duplicates=False)       
                    for s in range(len(for_x)):#填入內容                    
                        for_x['CEIC分類'][s]=for_x['index'][s][0]
                        for_x['變數名稱'][s]=for_x['index'][s][1]  
                    for_x = for_x.drop('index',axis=1)
                df_for_x = pd.concat([df_for_x,for_x],axis=1)   
                print(name)
                x_data = self.X[self.result['h='+str(h)][name][i+2].index]
                #新增欄
                col_name=x_data.columns.tolist()  # 將數據框的列名全部提取出來存放在列表裡           
                col_name.insert(0,name2)   # 在列索引為0的位置插入一列,列名為name，剛插入時不會有值，整列都是NaN                  
                x_data=x_data.reindex(columns=col_name) # DataFrame.reindex() 對原行/列索引重新構建索引值            
                df_for_x_data = pd.concat([df_for_x_data,x_data],axis=1)
                    
                df_for_x.to_excel(writer, sheet_name='(h='+str(h)+')'+' x_coef_corr',index=False) #coef最負與最正 & corr最負與最正        
                df_for_x_data.to_excel(writer, sheet_name='(h='+str(h)+')'+' x_data')        
    
            self.result['h='+str(h)][name][-1].to_excel(writer, sheet_name='mse')
            self.mse = self.result['h='+str(h)][name][-1].values
            writer.save() # 存檔生成excel檔案





















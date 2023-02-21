#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 16:19:08 2022

@author: kian
"""

import numpy as np
import pandas as pd
import seaborn as sns; sns.set()


#------------- Specifications -------------------------------------------------
path0 = '/home/kian/Dropbox/NTPU/RA_project/RA/Janice/臺經院計畫/長興中國大陸經營情勢指標資料/長興中國大陸經營情勢指標資料/code' # Working directory
path = '/home/kian/Dropbox/NTPU/RA_project/RA/Janice/臺經院計畫/長興中國大陸經營情勢指標資料/長興中國大陸經營情勢指標資料/' # Working directory
#path = r'/home/kian/Dropbox/DjangoProject_byJanice/mysite/myapp2/' # Working directory
#path = r'C:\Users\kian_\Dropbox\NTPU\RA_project\RA\Janice\臺經院計畫\長興中國大陸經營情勢指標資料\長興中國大陸經營情勢指標資料\code'


import os # Janice 新加入
os.chdir(path0) # Janice 新加入
from Eternal_project import * # Janice 調整code位置

# gridsearch parameters
grid = dict()
grid['n_estimators'] = [10, 50, 100] # ABR, GBR, BR
grid['learning_rate'] = [0.001, 0.01, 0.1, 1.0] # ABR, GBR
grid['subsample'] = [0.7, 1.0] #GBR
grid['max_samples'] = [0.7, 1.0] # BR
grid['max_depth'] = [1,2,3] # GBR


h0 = [9]
ass = 'covid'
dataset = 'all' # all or 20 or long
### Spcification of in-sample and out-of-sample 
if ass == 'covid':
    tt0 = pd.Timestamp('2020-1-1',freq='MS')
    tt1 = pd.Timestamp('2021-12-1',freq='MS')
elif ass == 'normal':
    tt0 = pd.Timestamp('2017-1-1',freq='MS')
    tt1 = pd.Timestamp('2020-12-1',freq='MS')
    
y_list= ['塗料產業_營收金額(台幣)','膠黏劑產業_營收金額(台幣)','PCB產業_營收金額(台幣)','塗料產業_出產數量(KG)','膠黏劑產業_出產數量_KG','PCB產業_出產數量_FT2(面積單位)']
#------------- Specifications -------------------------------------------------    
   

#------------- Load data ------------------------------------------------------
#data =  pd.read_excel(path+r'/CEIC資料庫資料與長興材料公司原物料均價.xlsx',sheet_name=None,skiprows=[1], parse_dates={'time': [0]}, date_parser=my_date_parser, index_col='time')
data0 =  pd.read_excel('/home/kian/Dropbox/NTPU/CEIC資料庫資料與長興材料公司原物料均價0209.xlsx',sheet_name=None,skiprows=[1], parse_dates={'time': [0]}, date_parser=my_date_parser, index_col='time')
#y = pd.read_excel(path+r'/長興材料公司_1998年至2021年_各產業營收金額與銷售量 .xlsx',sheet_name=None,skiprows=[1], parse_dates={'time': [0]}, date_parser=my_date_parser, index_col='time')
y = pd.read_excel('/home/kian/Dropbox/NTPU/長興2015-2022三大產業銷售金額與數量.xlsx',sheet_name=None,skiprows=[1], parse_dates={'time': [0]}, date_parser=my_date_parser, index_col='time')
cat_list = list(data.keys())
#------------- Load data ------------------------------------------------------


#------------- Pre-process of y -----------------------------------------------
y1 = y['銷售量與營收']
y1 = y1.replace({0:np.nan})
y1rolling = y1.rolling(12, min_periods=1, center=False).mean()
y1.update(y1rolling, overwrite=False)
y1 = y1.interpolate(limit_area='inside')
y1 = y1.pct_change(12)
y1.index = pd.date_range(y1.index[0],y1.index[-1],freq='MS')
y1,n = remove_outliers(y1,15)
y1 = y1.interpolate(limit_area='inside')
#------------- Pre-process of y -----------------------------------------------




#------------- Main process ---------------------------------------------------

for ii in y_list:
    data = data0.copy()
    if ii.find('膠黏劑') == 0:
        del data['20大原料單價(台幣均價_計量單位全為KG）']
        del data['PCB']
        del data['塗料']
    elif ii.find('PCB') == 0:
        del data['20大原料單價(台幣均價_計量單位全為KG）']
        del data['膠粘劑']
        del data['塗料']
    elif ii.find('塗料') == 0:
        del data['20大原料單價(台幣均價_計量單位全為KG）']
        del data['PCB']
        del data['膠粘劑']
    model = Eternal_project(y1,data,path,h0,grid,ass)
    model.dataset_X(dataset)
    model.est(ii)
    model.importances(9,ii)
    result = model.fcast(ii)
    model.plot('importance')
    model.plot('fcast')
    model.to_excel(ii)    
#------------- Main process ---------------------------------------------------


#==== Janice 新加入，此處目的在於得到這個月的 4 個 piclle 檔案 ==========
#path = r'/home/kian/Desktop/test' # Working directory
for ass in ['covid','normal']:
    if ass == 'covid':
        tt0 = pd.Timestamp('2020-1-1',freq='MS')
        tt1 = pd.Timestamp('2021-12-1',freq='MS')
    elif ass == 'normal':
        tt0 = pd.Timestamp('2017-1-1',freq='MS')
        tt1 = pd.Timestamp('2020-12-1',freq='MS')
    for dataset in ['all','20']:
        model = Eternal_project(y1,data,path,h0,grid,tt0,tt1,ass)
        model.dataset_X(dataset)
        globals()['est_'+ass+'_'+dataset] = model.est() #變數名稱跟著迴圈動態定義(全域變數)
        
        
#==== Janice 新加入，此處僅測試能否成功讀取  piclle 檔案 ==========
        

#scenario = 'covid' #ass in ['covid','normal']
#dataset = 'all'#dataset in ['all','20']
h=6
import os # Janice 加入
import pickle
for ass in ['covid','normal']:
    for dataset in ['all','20']:
        file_path = os.path.join(path, '202301', dataset, ass) # Janice 修改過的地方
        try:
            with open(file_path + '/estimators.pickle', 'rb') as f:
                print('est_'+ass+'_'+dataset)
                
                globals()['est_'+ass+'_'+dataset] = pickle.load(f)
                print('讀取成功',f)
                print('＝＝＝＝＝＝＝＝＝＝')
        except EOFError:
            print('讀取失敗(文件為空)',f)

#==== Janice 亂測試 ==========
name = y_list[0]
h = h0[0]
for scenario in ['covid','normal']:
    for dataset in ['all','20']:
        print('est_'+scenario+'_'+dataset)
        estimators = globals()['est_'+scenario+'_'+dataset]
        print(len(estimators['h='+str(h)][name][4].coef_))
        print('＝＝＝＝＝＝＝＝＝＝')



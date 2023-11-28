# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 15:36:16 2020

@author: mayson
"""

#%%

import os
os.chdir(r'C:\Users\dpelt\Desktop\Mayson\UOS_graduate\kmooc\ppt')
print('current directory:', os.getcwd())

#%%

'''4주차 1,2차시'''

# 행렬을 다루기 위한 모듈을 불러온다.
import numpy as np
# 데이터를 읽어오기 위한 pandas 모듈을 불러온다.
import pandas as pd 
# 선형회귀모형을 사용할 수 있는 모듈을 불러온다.
import statsmodels.api as sm
import statsmodels.formula.api as smf

# csv 파일 형식으로 되어있는 Default 데이터를 불러온다.
# 데이터: http://faculty.marshall.usc.edu/gareth-james/ISL/data.html
default = pd.read_csv('Default.csv', encoding='cp949')
# 필요한 열만을 추출
default = default[['default', 'balance']]
# 데이터 확인
default.head()

#%%

# 모형 적합에 사용되는 predictor에 상수항을 추가
x = default['balance']
x = sm.add_constant(x)
x.iloc[:6, :]

# response variable인 default 변수를 dummy(0, 1) 변수 코딩
y = list(map(lambda x:1 if x == 'Yes' else 0, default['default']))
y[:6]

#%%

# 선형 회귀모형 적합
linear_reg = sm.OLS(y, x).fit()
# 적합 결과 확인
linear_reg.summary()

#%%

# 적합된 선형 회귀모형과 response variable의 시각화
import matplotlib.pyplot as plt
linear_reg_pred = linear_reg.predict(x)
plt.plot(default['balance'], linear_reg_pred)
plt.plot(default['balance'], default['default'], 
         linestyle='none', marker='o', markersize=2, color='red')

#%%

# logistic 회귀모형 적합
logistic_reg = sm.Logit(y, x).fit()
# 적합 결과 확인
logistic_reg.summary()

#%%

# 적합된 logistic 회귀모형과 response variable의 시각화
logistic_reg_pred = logistic_reg.predict(x)
plt.plot(np.sort(default['balance']), np.sort(logistic_reg_pred))
plt.plot(default['balance'], default['default'], 
         linestyle='none', marker='o', markersize=2, color='red')

#%%

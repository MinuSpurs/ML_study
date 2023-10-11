#%%

'''6주차 1차시'''

# 행렬을 다루기 위한 모듈을 불러온다.
import numpy as np
# 데이터를 읽어오기 위한 pandas 모듈을 불러온다.
import pandas as pd 
# 선형회귀모형을 사용할 수 있는 모듈을 불러온다.
import statsmodels.api as sm

#%%

# csv 파일 형식으로 되어있는 advertising 데이터를 불러온다.
# 데이터: http://faculty.marshall.usc.edu/gareth-james/ISL/data.html
advertising = pd.read_csv('Advertising.csv', encoding='cp949', index_col=0) 

# 데이터의 개수
len(advertising)

# train, validation, test set의 분할
train = advertising[:100]
train.head()
val = advertising[50:150]
val.head()
test = advertising[150:]
test.head()

#%%

# Bestsubset Selection에서 사용될 predictor들의 조합을 생성
import itertools
from copy import deepcopy
predictors = ['TV', 'radio', 'newspaper']
bestsubset = deepcopy(predictors)
for i in range(2, len(predictors)+1):
    bestsubset.extend([list(x) for x in list(itertools.combinations(predictors, i))])
bestsubset

#%%

# 7개의 모형의 subset들에 대해서 모형을 적합하고
# validation set에 대한 MSE를 계산
val_mse = []
for i in range(len(bestsubset)):
    simple_reg = sm.OLS(train['sales'], train[bestsubset[i]]).fit()
    simple_reg_pred = np.array(simple_reg.predict(val[bestsubset[i]]))
    mse = (((simple_reg_pred - np.array(val['sales'])) ** 2).sum()) / len(simple_reg_pred)
    val_mse.append(mse)
val_mse

#%%

# 가장 작은 validation MSE를 가지는 subset을 선택
np.argsort(val_mse)[0]
min_mse_subset = bestsubset[np.argsort(val_mse)[0]]
min_mse_subset 

# 앞에서 고른 bestsubset를 이용하여 test 데이터에 대한 MSE를 계산
best_subset_reg = sm.OLS(train['sales'], train[min_mse_subset]).fit()
best_subset_reg_pred = np.array(best_subset_reg.predict(test[min_mse_subset]))
test_mse = (((best_subset_reg_pred - np.array(test['sales'])) ** 2).sum()) / len(best_subset_reg_pred)
test_mse 

#%%

'''6주차 2차시'''

# csv 파일 형식으로 되어있는 advertising 데이터를 불러온다.
# 데이터: http://faculty.marshall.usc.edu/gareth-james/ISL/data.html
advertising = pd.read_csv('Advertising.csv', encoding='cp949', index_col=0) 

# 5-fold cross validation
k = 5
# 각 validation set의 크기
val_size = int(len(advertising) / k)
val_size
predictors = ['TV', 'radio', 'newspaper']

#%%

# 1개의 validation set을 분할
# -> 관측치들의 번호를 분할
i = 0
idx = np.arange(len(advertising))
cv_val_idx = np.arange(i*val_size, (i+1)*val_size)
cv_val_idx 
cv_train_idx = np.array([x for x in idx if x not in cv_val_idx])
cv_train_idx 

#%%

# 1개의 cross-validation set을 이용하여 MSE를 계산
reg = sm.OLS(advertising['sales'].iloc[cv_train_idx], # train set을 이용
             advertising[predictors].iloc[cv_train_idx]).fit()
cv_pred = np.array(reg.predict(advertising[predictors].iloc[cv_val_idx]))
mse = (((cv_pred - np.array(advertising['sales'].iloc[cv_val_idx])) ** 2).sum()) / len(cv_pred)
mse 

#%%

# 모든 5개의 cross-validation set에 대해서 MSE를 계산
cv_mse = []
for i in range(k):
    idx = np.arange(len(advertising))
    cv_val_idx = np.arange(i*val_size, (i+1)*val_size)
    cv_train_idx = np.array([x for x in idx if x not in cv_val_idx])
    reg = sm.OLS(advertising['sales'].iloc[cv_train_idx], 
                 advertising[predictors].iloc[cv_train_idx]).fit()
    cv_pred = np.array(reg.predict(advertising[predictors].iloc[cv_val_idx]))
    mse = (((cv_pred - np.array(advertising['sales'].iloc[cv_val_idx])) ** 2).sum()) / len(cv_pred)
    cv_mse.append(mse)

# 각 cross-validation set들의 MSE 결과
cv_mse
# 5개의 cross-validation MSE의 평균
np.sum(cv_mse) / k

#%%

#%%
import os
os.chdir(r'C:\Users\dpelt\Desktop\Mayson\UOS_graduate\kmooc\ppt\week2')
print('current directory:', os.getcwd())
#%%

# 행렬을 다루기 위한 모듈을 불러온다.
import numpy as np
# 선형회귀모형을 사용할 수 있는 모듈을 불러온다.
import statsmodels.api as sm

# 관측치의 개수
N = 1000
# 정규분포를 이용해 데이터를 생성(N(0, 1) 분포)
X = np.random.normal(loc=0, scale=1, size=(N, 1))
# 절편(상수항)을 추가
X = sm.add_constant(X)
X[:6, :]

#%%

# 정규분포를 이용해 오차항을 생성(N(0, 1) 분포)
epsilon = np.random.normal(loc=0, scale=1, size=(N, 1))
# 실제 모형: Y = 2 + 3X + epsilon
Y = X @ np.array([[2], 
                  [3]]) + epsilon
Y[:6, :]

#%%

# 단순회귀모형 적합
simple_reg = sm.OLS(Y, X).fit()
# 적합 결과 확인
simple_reg.summary()

#%%

# 관측치의 개수
N = 1000
# 정규분포를 이용해 데이터를 생성(N(0, 1) 분포)
X = np.random.normal(loc=0, scale=1, size=(N, 1))
# 절편(상수항)을 추가
X = sm.add_constant(X)

# 정규분포를 이용해 오차항을 생성(N(0, 1) 분포)
epsilon = np.random.normal(loc=0, scale=1, size=(N, 1))
# 실제 모형: Y = 2 + 3X + epsilon
Y = X @ np.array([[2], 
                  [3]]) + epsilon

# 단순회귀모형 적합
simple_reg = sm.OLS(Y, X).fit()
# 적합 결과 확인
simple_reg.summary()

#%%

# 관측치의 개수
N = 1000
# 모형 적합을 반복하는 횟수
repeat_num = 1000

beta = np.zeros((2, ))
for k in range(repeat_num):
    # 정규분포를 이용해 데이터를 생성(N(0, 1) 분포)
    X = np.random.normal(loc=0, scale=1, size=(N, 1))   
    # 절편(상수항)을 추가
    X = sm.add_constant(X)
    # 정규분포를 이용해 오차항을 생성(N(0, 1) 분포)
    epsilon = np.random.normal(loc=0, scale=1, size=(N, 1))
    # 실제 모형: Y = 2 + 3X + epsilon
    Y = X @ np.array([[2], [3]]) + epsilon
    
    simple_reg = sm.OLS(Y, X).fit()
    beta += simple_reg.params
    
# 적합된 회귀 계수의 평균을 계산
beta /= repeat_num
beta

#%%

# 데이터를 읽어오기 위한 pandas 모듈을 불러온다.
import pandas as pd 

# csv 파일 형식으로 되어있는 mtcars 데이터를 불러온다.
# 데이터: https://www.statlearning.com/resources-first-edition
advertising = pd.read_csv('advertising.csv', encoding='cp949', index_col=0)
# TV 변수를 선택, numpy 행렬로 변환
X = np.array(advertising[['TV']])
# 절편(상수항)을 추가
X = sm.add_constant(X)
X[:6, :]

#%%

# response variable인 'sales'를 numpy 행렬로 변환
Y = np.array(advertising[['sales']])
Y[:6, :]

# 단순회귀모형 적합
simple_reg = sm.OLS(Y, X).fit()
# 적합 결과 확인
simple_reg.summary()

#%%

# pandas 모듈을 불러온다.
import pandas as pd 

# csv 파일 형식으로 되어있는 mtcars 데이터를 불러온다.
advertising = pd.read_csv('advertising.csv', encoding='cp949', index_col=0) 
# 모든 설명 변수를 선택, numpy 행렬로 변환
X = np.array(advertising[['TV', 'radio', 'newspaper']])
# 절편(상수항)을 추가
X = sm.add_constant(X)
X[:6, :]

#%%

# response variable인 'sales'를 numpy 행렬로 변환
Y = np.array(advertising[['sales']])
Y[:6, :]

# 다변량회귀모형 적합
multi_reg = sm.OLS(Y, X).fit()
# 적합 결과 확인
multi_reg.summary()

#%%

# 관측치의 개수
N = 1000
# 정규분포를 이용해 데이터를 생성(N(0, 1) 표준정규분포)
X = np.random.normal(loc=0, scale=1, size=(N, 4))
X[:6, :]

# 실제 회귀 계수
beta = np.array([[0], [3], [1], [0]])
beta

#%%

# 정규분포를 이용해 오차항을 생성(N(0, 1) 표준정규분포)
epsilon = np.random.normal(loc=0, scale=1, size=(N, 1))
# 참 모형: Y = 0*X1 + 3*X2 + 1*X3 + 0*X4 + epsilon
Y = X @ beta + epsilon
Y[:6, :]

#%%


# 평가할 후보 변수
candidates = list(range(4))
candidates

# step. 1
bic = []
for predictor in candidates:
    # 회귀모형 적합
    reg = sm.OLS(Y, X[:, predictor]).fit()
    # BIC 값 저장
    bic.append(reg.bic)
bic

#%%

# BIC 값이 가장 작은 변수를 선택
chosen1 = np.argmin(bic)
chosen1 

# 선택한 변수를 후보에서 제외
del candidates[chosen1]
candidates

#%%

# step. 2
bic = []
for predictor in candidates:
    # 선택한 설명 변수를 포함하여 회귀모형 적합
    reg = sm.OLS(Y, X[:, [chosen1, predictor]]).fit()
    # BIC 값 저장
    bic.append(reg.bic)
bic

#%%

# BIC 값이 가장 작은 변수를 선택
chosen2 = np.argmin(bic)
chosen2

# 선택한 변수를 후보에서 제외
del candidates[chosen2]
candidates

#%%

# step. 3
bic = []
for predictor in candidates:
    # 선택한 설명 변수를 포함하여 회귀모형 적합
    reg = sm.OLS(Y, X[:, [chosen1, chosen2, predictor]]).fit()
    # BIC 값 저장
    bic.append(reg.bic)
bic

#%%















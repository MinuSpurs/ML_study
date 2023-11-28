#%%

'''11주차'''
 
# 모형 적합에 사용할 데이터 생성
import numpy as np
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
np.random.seed(1)
n = 100
y = np.random.normal(0, 1, n)
y[20:30] += 7
y[60:80] += 14

# 시각화
plt.figure(figsize=(6, 3))
plt.plot(y, c='b')

#%%

# beta를 delta로 바꾸기위해 필요한 행렬
D = np.tril(np.ones((n, n-1)), k=-1)
D.shape
D

#%%

# 특정한 alpha의 값에서 Fused Lasso 모형 적합과 delta 확인
a = 0.1
lasso = Lasso(alpha=a, fit_intercept=True)
lasso.fit(D, y)
delta = lasso.coef_
delta.shape
delta[:20]
lasso.intercept_

#%%

# delta를 beta로 변환
beta_hat = np.cumsum(delta) + lasso.intercept_
beta_hat[:20]

#%%

# 시각화
plt.figure(figsize=(6, 3))
plt.plot(y, c='b')
plt.plot(beta_hat, c='r', linestyle='--')
plt.axis('tight')

#%%

n = 100
true_trend = np.linspace(0,5,int(n/2))
true_trend = np.append(true_trend, np.linspace(5,0,int(n/2)))
# 시각화
plt.figure(figsize=(6, 3))
plt.plot(true_trend, c='b')

#%%    

# 관측된 데이터 생성
np.random.seed(1000)
y = true_trend + np.random.normal(0, 1, n)
y = y.reshape((n, 1))
y[:10, :]
# 시각화
plt.figure(figsize=(6, 3))
plt.plot(y, c='b')  
    
#%%

# parameter 설정
rho = 1
l = 5
D = (np.eye(n) + np.diag([-2]*(n-1), k=1) + np.diag([1]*(n-2), k=2))[:-2, :]
D.shape
D
# initial value
beta = np.ones((n ,1))
z = np.ones((n-2 ,1))
mu = np.ones((n-2, 1))

#%%

# ADMM의 일부분
# beta update
beta = np.linalg.inv((rho/2) * D.T @ D + np.eye(n)) @ (y - D.T @ (mu - rho*z)/2)
beta[:6, :]

#%%

# z update
z_temp = D @ beta + mu/rho
for i in range(n-2):
    if z_temp[i] < -l/rho:
        z[i] = z_temp[i] + l/rho
    elif z_temp[i] > l/rho:
        z[i] = z_temp[i] - l/rho
    else:
        z[i] = 0
z[:6, :]

#%%

# mu update
mu = mu + rho*(D @ beta - z)
mu[:6, :]

#%%

# parameter 설정
rho = 1
l = 5
D = (np.eye(n) + np.diag([-2]*(n-1), k=1) + np.diag([1]*(n-2), k=2))[:-2, :]
# initial value
beta = np.ones((n ,1))
z = np.ones((n-2 ,1))
mu = np.ones((n-2, 1))
# ADMM
for k in range(2000):
    # beta update
    beta = np.linalg.inv((rho/2) * D.T @ D + np.eye(n)) @ (y - D.T @ (mu - rho*z)/2)
    # z update
    z_temp = D @ beta + mu/rho
    for i in range(n-2):
        if z_temp[i] < -l/rho:
            z[i] = z_temp[i] + l/rho
        elif z_temp[i] > l/rho:
            z[i] = z_temp[i] - l/rho
        else:
            z[i] = 0
    # mu update
    mu = mu + rho*(D @ beta - z)      
    
#%%

# 시각화
plt.figure(figsize=(6, 3))
plt.plot(y, c='b')
plt.plot(beta, c='r', linestyle='--', linewidth=3)
plt.axis('tight')
    
#%%

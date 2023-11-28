#%%

'''9주차'''

# Ridge penalty를 적용한 로지스틱 회귀분석

# 행렬을 다루기 위한 모듈을 불러온다.
# no interceptor model
import numpy as np
np.random.seed(1)
p = 2
lambda_v = 1.5
true_beta = np.array([[1], [-0.5]])
n = 10

# predictor 생성
x = np.random.normal(0, 1, (n, p))
# 확률값을 계산하고 binomial 분포를 이용해 y를 생성
prob = 1 / (1 + np.exp(- x @ true_beta))
prob = prob.reshape((n,))
prob 

tmpN = np.ones(n, dtype='int32')
y = np.random.binomial(tmpN, prob, n)
y = y.reshape((n,1))
y

#%%

beta = np.array([.5,.5]).reshape((p,1))
prob = 1 / (1 + np.exp(- x @ beta))
# 1차 미분
grad = np.mean((prob - y) * x, axis=0, keepdims=True).T + 2*lambda_v*beta
# 2차 미분
D = np.diag((prob * (1 - prob)).reshape(n))
D[:5, :5]
H = x.T @ D @ x/n + np.diag(np.repeat(2*lambda_v,p))
H

#%%

# initial beta
beta = np.zeros((p, 1))
for i in range(10):
    # 확률, 1차와 2차 미분 계산
    prob = 1 / (1 + np.exp(- x @ beta))
    grad = np.mean((prob - y) * x, axis=0, keepdims=True).T + 2*lambda_v*beta
    D = np.diag((prob * (1 - prob)).reshape(n))
    H = x.T @ D @ x/n + np.diag(np.repeat(2*lambda_v,p))
    
    # beta update
    beta_new = beta - np.linalg.inv(H) @ grad
    
    # stopping rule
    if np.sum(np.abs(beta_new - beta)) < 1e-8:
        beta = beta_new
        print('Iteration {} beta:'.format(i+1))
        print(beta, '\n')
        break
    else:
        beta = beta_new
        print('Iteration {} beta:'.format(i+1))
        print(beta, '\n')

#%%
        
# KKT condition
prob = 1 / (1 + np.exp(- x @ beta))
grad = np.mean((prob - y) * x, axis=0, keepdims=True).T + 2*lambda_v*beta
np.all(np.abs(grad) < 1e-8) # stationarity

#%%
        
# lasso, ridge linear regression
# 행렬을 다루기 위한 모듈을 불러온다.
import numpy as np
# 시각화를 위한 모듈을 불러온다.
import matplotlib.pyplot as plt
# Ridge와 Lasso를 사용하기위한 모듈을 불러온다.
from sklearn.linear_model import lasso_path, Lasso, Ridge

# 조율모수(lambda_v) 값 설정
n_lambdas = 50
lambda_vec = np.linspace(0,100, n_lambdas)
# 각 alpha 값에 대하여 Ridge penalty regression 적합(solution path)
coefs = []
for lambda_v in lambda_vec:
    ridge = Ridge(alpha=lambda_v, fit_intercept=False)
    ridge.fit(x, y)
    coefs.append(ridge.coef_)
coefs = np.squeeze(np.array(coefs))
coefs[:6, :]

#%%

# solution path 시각화
plt.figure(figsize=(6, 3))
colors = ['b', 'r', 'g']
lstyles = ['-', '--', '-.', ':']
for i in range(p):
    l = plt.plot(lambda_vec, coefs[:, i], 
                 linestyle=lstyles[i], c=colors[i])
plt.xscale('log')
plt.axis('tight')
plt.show()

#%%

# 특정 alpha에서 beta의 값 확인
lambda_v = 10
ridge = Ridge(alpha=lambda_v, fit_intercept=False)
ridge.fit(x, y)
ridge.coef_

#%%

# Lasso penalty regression의 solution path
eps = 5e-3
lambdas_lasso, coefs_lasso, _ = lasso_path(x, y, eps=eps, fit_intercept=False)
coefs_lasso = np.squeeze(coefs_lasso)
coefs_lasso[:, :5]

#%%

# solution path 시각화
plt.figure(figsize=(6, 3))
colors = ['b', 'r', 'g']
lstyles = ['-', '--', '-.', ':']
for coef_l, c, ltype in zip(coefs_lasso, colors, lstyles):
    l = plt.plot(lambdas_lasso, coef_l, 
                 linestyle=ltype, c=c)
plt.axis('tight')

#%%
    
# 특정 alpha에서 beta의 값 확인
lambda_v = 0.01
lasso = Lasso(alpha=lambda_v, fit_intercept=False)
lasso.fit(x, y)
lasso.coef_

#%%

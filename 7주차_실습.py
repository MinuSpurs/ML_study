#%%

# poisson regression model

# 행렬을 다루기 위한 모듈을 불러온다.
import numpy as np
np.random.seed(1)
p = 2
true_beta = np.array([[1], [0.5]])
n = 1000
x = np.random.normal(loc=0, scale=0.2, size=(n, 1))
x = np.hstack((np.ones((n, 1)), x))
x[:5, :]
# 모수값을 계산하고 poisson 분포를 이용해 y를 생성
parm = np.exp(x @ true_beta)
parm[:5, :]
y = np.random.poisson(parm)
y[:5, :]

#%%

'''gradient descent'''

beta = np.array([.5, .5]).reshape((p,1))
# 1차 미분
parm = np.exp(x @ beta)
grad = -np.mean(y*x - parm*x, axis=0).reshape((p,1))
grad

#%%

learning_rate = 0.7
# initial beta
beta = np.zeros((p, 1))
for i in range(500):
    # 모수추정값, 1차 미분 계산
    parm = np.exp(x @ beta)
    grad = -np.mean(y*x - parm*x, axis=0).reshape((p,1))

    # beta update
    beta_new = beta - learning_rate*grad
    
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

'''Newton-Raphson'''

beta = np.array([.5, .5]).reshape((p,1))
# 2차 미분
parm = np.exp(x @ beta)
D = np.diag(np.squeeze(parm))
D
H = x.T @ D @ x/n
H

#%%

# initial beta
beta = np.zeros((p, 1))
for i in range(500):
    # 모수추정값, 1차와 2차 미분 계산
    parm = np.exp(x @ beta)
    grad = -np.mean(y*x - parm*x, axis=0).reshape((p,1))
    D = np.diag(np.squeeze(parm))
    H = x.T @ D @ x/n
    
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

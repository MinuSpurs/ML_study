#%%

'''10주차'''

# 행렬을 다루기 위한 모듈을 불러온다.
import numpy as np
# 시각화를 위한 모듈을 불러온다.
import matplotlib.pyplot as plt
# Lasso를 사용하기위한 모듈을 불러온다.
from sklearn.linear_model import lasso_path

# 모형 적합에 사용할 predictor 생성
np.random.seed(1)
n = 1000
x1 = np.random.normal(0, 1, (n, 1))
x2 = np.random.normal(0, 1, (n, 1))
e = np.random.normal(0, 1, (n, 1))
x3 = (2/3)*x1 + (2/3)*x2 + (1/3)*e
x = np.hstack((x1, x2, x3))
x[:6, :]

#%%

# case 1.
beta1 = np.array([[2], [3]])
# 모형 적합에 사용할 response variable 생성
y = np.hstack((x1, x2)) @ beta1 + np.random.normal(0, 1, (n, 1))
y[:6, :]
# Lasso penalty regression의 solution path
eps = 5e-3
lambdas_lasso, coefs_lasso, _ = lasso_path(x, y, 
                                          eps=eps, fit_intercept=False)
coefs_lasso.shape
coefs_lasso = np.squeeze(coefs_lasso)
coefs_lasso[:,20]

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

# case 2.
beta2 = np.array([[-2], [3]])
# 모형 적합에 사용할 response variable 생성
y = np.hstack((x1, x2)) @ beta2 + np.random.normal(0, 1, (n, 1))
y[:6, :]
# Lasso penalty regression의 solution path
eps = 5e-3
lambdas_lasso, coefs_lasso, _ = lasso_path(x, y, 
                                          eps=eps, fit_intercept=False)
coefs_lasso = np.squeeze(coefs_lasso)
coefs_lasso.shape
coefs_lasso[:,20]

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
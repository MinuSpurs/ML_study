#%%

'''5주차 1차시'''

# 행렬을 다루기 위한 모듈을 불러온다.
import numpy as np

import matplotlib.pyplot as plt

#%%

mu0 = [0, 0]
sigma0 = np.array([[1, 0.5], [0.5, 1]])
sigma1 = np.array([[1, 0.5], [0.5, 1]])
# 데이터 생성
x = np.random.multivariate_normal(mu0, sigma0, 5000) 
x[2000:, 0] = x[2000:, 0] + 2
x[2000:, 1] = x[2000:, 1] + 2
y = np.ones((5000, ))
# 처음 2000개의 데이터는 label을 0으로 지정
y[:2000] = 0

#%%

# 데이터 확인
x[:6, :]
y[:6]

mu0 = np.array([[0], [0]])
mu0
mu1 = np.array([[5], [5]])
mu1
sigma0
sigma1

#%%

# 데이터 시각화
plt.scatter(x[:, 0], x[:, 1], 
            s=3, c=list(map(lambda x:'blue' if x==0 else 'red', y)))

#%%

# label 0에 해당하는 임의의 sample 데이터
sample_x = np.array([[0.5], [0.5]])

# 실제 모수를 이용하여 계산한 pdf 값
def f_x_y0(x): # 수식 12
    const = 1 / (2*np.pi * np.power(np.linalg.det(sigma0), 1/2))
    return const * np.exp(-0.5 * (x - mu0).T @ np.linalg.inv(sigma0) @ (x - mu0))[0, 0]

def f_x_y1(x): # 수식 13
    const = 1 / (2*np.pi * np.power(np.linalg.det(sigma1), 1/2))
    return const * np.exp(-0.5 * (x - mu1).T @ np.linalg.inv(sigma1) @ (x - mu1))[0, 0]

f_x_y0(sample_x)
f_x_y1(sample_x)

#%%

# 모수에 대한 추정

mu0_hat = np.mean(x[:2000, :], axis=0)
mu0_hat
sigma0_hat = np.cov(x[:2000, :].T)
sigma0_hat 

mu1_hat = np.mean(x[2000:, :], axis=0)
mu1_hat 
sigma1_hat = np.cov(x[2000:, :].T)
sigma1_hat 

#%%

# label 0에 해당하는 임의의 sample 데이터
sample_x = np.array([[0.5], [0.5]])

# 추정된 모수를 이용하여 계산한 pdf 값
def fhat_x_y0(x): # 수식 12
    const = 1 / (2*np.pi * np.power(np.linalg.det(sigma0_hat), 1/2))
    return const * np.exp(-0.5 * (x - mu0_hat).T @ np.linalg.inv(sigma0_hat) @ (x - mu0_hat))[0, 0]

def fhat_x_y1(x): # 수식 13
    const = 1 / (2*np.pi * np.power(np.linalg.det(sigma1_hat), 1/2))
    return const * np.exp(-0.5 * (x - mu1_hat).T @ np.linalg.inv(sigma1_hat) @ (x - mu1_hat))[0, 0]

fhat_x_y0(sample_x)
fhat_x_y1(sample_x)

#%%

# 각 label에 대한 실제 prior 확률
p0 = 2000 / 5000
p1 = 3000 / 5000

def bayse_prob(x, y): # 수식 11
    if y == 1:
        return f_x_y1(x)*p1 / (f_x_y0(x)*p0 + f_x_y1(x)*p1)
    else:
        return f_x_y0(x)*p0 / (f_x_y0(x)*p0 + f_x_y1(x)*p1)

bayse_prob(sample_x, 0)
bayse_prob(sample_x, 1)
    
#%%

# 각 label에 대한 추정된 prior 확률
p0_hat = 2000 / 5000
p1_hat = 3000 / 5000
    
def bayse_prob_hat(x, y): # 수식 11
    if y == 1:
        return fhat_x_y1(x)*p1_hat / (fhat_x_y0(x)*p0_hat + fhat_x_y1(x)*p1_hat)
    else:
        return fhat_x_y0(x)*p0_hat / (fhat_x_y0(x)*p0_hat + fhat_x_y1(x)*p1_hat)

bayse_prob_hat(sample_x, 0)
bayse_prob_hat(sample_x, 1)

#%%

'''5주차 2차시'''

# LDA를 사용하기 위한 모듈을 불러온다
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# LDA 모형을 적합
lda = LinearDiscriminantAnalysis(n_components=1, solver="svd", store_covariance=True).fit(x, y)
# 추정된 모수를 확인
lda_means = lda.means_
mu0 = lda_means[0, :]
mu0
mu1 = lda_means[1, :]
mu1
lda_cov = lda.covariance_
lda_cov 

#%%

# 각 label에 대한 prior 확률
pi0 = 2000 / 5000
pi1 = 3000 / 5000

# 판별식
def discriminator(x): # 수식 8
    d0 = x.T @ np.linalg.inv(lda_cov) @ mu0 - 0.5 * mu0.T @ np.linalg.inv(lda_cov) @ mu0 + np.log(pi0)
    d1 = x.T @ np.linalg.inv(lda_cov) @ mu1 - 0.5 * mu1.T @ np.linalg.inv(lda_cov) @ mu1 + np.log(pi1)
    return [d0[0], d1[0]]

# 판별식 계산
discriminator(sample_x)
np.argmax(discriminator(sample_x))

#%%

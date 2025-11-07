import pandas as pd
import numpy as np

from parameters import *

# MFE5130 2025 奇异期权定价 二项式模型版本

#===生成二叉树列索引矩阵===
mu = np.arange(M + 1)
mu = np.resize(mu, (M + 1, M + 1))
md = np.transpose(mu)
#print("二叉树列索引矩阵md:\n", md)
mu = u ** (mu - md)
md = d ** md 
S = S0 * mu * md

#===计算内层期权价值矩阵===
iv = np.maximum(K2[5] - S ** 1.5, 0)  # 计算内层期权价值，欧式看涨期权
z = 0 
for j in range(M - 1, M-M_inner-1, -1):
    iv[0:M-z, j] = (p * iv[0:M - z, j + 1] + (1 - p) * iv[1:M - z + 1, j + 1]) * df
    z += 1
print("内层期权在T日的价格C(T):\n", iv[0, M - M_inner])
#===计算内层期权价值矩阵===

#===计算第M_outer步后的内层期权价格列===
iv[0:M_outer+1,M_outer] = np.maximum(N1*(iv[0:M_outer+1,M_outer] - K1),0)
#===计算第M_outer步后的内层期权价格列===


#===计算外层看涨期权价值矩阵===
z = 0
for i in range(M_outer,-1,-1):
    iv[0:M_outer-z,i] = (iv[0:M_outer-z,i+1]*p + iv[1:M_outer-z+1,i+1]*(1-p)) * df
    z +=1
print("外层期权在t日的价格C(t):\n", iv[0,0])
#===计算外层看涨期权价值矩阵===

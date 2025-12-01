import pandas as pd
import numpy as np

from parameters import *

# MFE5130 2025 奇异期权定价 二项式模型 
def binominal_priceX(style):
    #===生成二叉树列索引矩阵===
    mu = np.arange(M + 1)
    mu = np.resize(mu, (M + 1, M + 1))
    md = np.transpose(mu)
    #print("二叉树列索引矩阵md:\n", md)
    mu = u ** (mu - md)
    md = d ** md 
    S = S0 * mu * md

    iv = np.zeros_like(S)
    iv[:,M] = np.maximum(K2[5] - S[:,M]**1.5, 0)
    #===计算内层期权价值矩阵===
    #iv = np.maximum(K2[5] - S**1.5, 0)  # 计算内层期权价值，欧式看涨期权
    z = 0 
    for j in range(M - 1, M-M_inner-1, -1):
        holding_value = (p * iv[0:M - z, j + 1] + (1 - p) * iv[1:M - z + 1, j + 1]) * df
        if style == 'American':
            exercise_value = np.maximum(K2[5] - S[0:M-z,j]**1.5, 0)
            iv[0:M-z, j] = np.maximum(holding_value,exercise_value)
        elif style == 'European':
            iv[0:M-z, j] = holding_value
        z += 1
    #print("内层期权在T日的价格C(T):\n", iv[M - M_inner])
    #计算内层期权在T时刻的价格平均值和中位数以分析得出合理的K1
    # V1_T_mean = np.mean(iv[:,M_outer])
    # V1_T_median = np.median(iv[:,M_outer])
    # print(f"内层期权在T时刻的价格平均值为:{V1_T_mean},价格中位数为{V1_T_median}")
    # V1_t_mean = V1_T_mean * np.exp(-rf*C_outer)
    # V1_t_median = V1_T_median * np.exp(-rf*C_outer)
    # print(f"内层期权在t时刻的价格平均值为:{V1_t_mean},价格中位数为{V1_t_median}")
    #===计算内层期权价值矩阵===
    #===计算第M_outer步后的内层期权价格列===
    iv[0:M_outer+1,M_outer] = np.maximum(N1*(iv[0:M_outer+1,M_outer] - K1),0)
    #===计算第M_outer步后的内层期权价格列===


    #===计算外层看涨期权价值矩阵===
    z = 0
    for i in range(M_outer-1,-1,-1):
        iv[0:M_outer-z,i] = (iv[0:M_outer-z,i+1]*p + iv[1:M_outer-z+1,i+1]*(1-p)) * df
        z +=1
    #print("外层期权在t日的价格C(t):\n", iv[0,0])
    priceX = iv[0,0]
    #===计算外层看涨期权价值矩阵===
    return priceX

a = binominal_priceX('European')
b = binominal_priceX('American')
print(f"内层为欧式期权的X的二项式定价为:{a}")
print(f"内层为美式期权的X的二项式定价为:{b}")

print(f"欧式期权的价格变化率为{100*(a/48.43 - 1):.2f}%")
print(f"美式期权的价格变化率为{100*(b/51.18 - 1):.2f}%")

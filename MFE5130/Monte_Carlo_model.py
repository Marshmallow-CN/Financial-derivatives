import pandas as pd
import numpy as np

from tqdm import tqdm
from parameters import *
from utils import *
# MFE5130 2025 奇异期权定价 蒙特卡洛模型 欧式期权版本

#重新命名参数

def monte_carlo_priceX(style):
    #==生成N_outer个外层期权价格S_T==
    #创建一个随机数生成器rng
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(N_outer)
    muT = rf - q - 0.5 * volatility**2 
    sigT = volatility*np.sqrt(C_outer)
    S_T = S0*np.exp(muT*C_outer + sigT*Z) #形状 (N_outer,)

    # muC = rf - q - 0.5 * volatility**2 
    # sigC = volatility*np.sqrt(C_inner)
    # discount_C = np.exp(-rf*C_inner)
    discount_T = np.exp(-rf*C_outer)
    discounted_payoffs = np.zeros(N_outer)
    #V1_Ts = []
    #遍历每一个外层价格ST求这个ST对应的Px
    for j in tqdm(range(N_outer), desc='计算中'):
        s_t = S_T[j]
        #z_inner = rng.standard_normal(N_inner)
        if style=='European':
            # S_TC = s_t * np.exp(muC*C_inner + sigC * z_inner) #得到N_inner个T+C时刻的股价
            # #计算T+C时刻的N_inner个内层期权收益
            # payoff_inner = np.maximum(K2[5] - S_TC ** 1.5, 0)
            # #计算T时刻内层期权的价格
            # V1_T = payoff_inner.mean() * discount_C
            #V1_Ts.append(V1_T)
            params = {
                "S0": s_t,          # 初始股价 (Initial Stock Level)
                "K": K2[5],           # 行权价格 (Strike Price)
                "T": C_inner,           # 到期时间 (Time-to-Maturity)
                "r": rf,           # 无风险利率 (Short Rate)
                "sigma": volatility,       # 波动率 (Volatility)
                "q": q, # 股息收益率
                
                # --- 模拟参数 ---
                "I": N_inner,           # 模拟路径数量
            }
            V1_T = European_MC_calculation(params)
        elif style=='American':
            params = {
                # --- 模型参数 ---
                "S0": s_t,          # 初始股价 (Initial Stock Level)
                "K": K2[5],           # 行权价格 (Strike Price)
                "T": C_inner,           # 到期时间 (Time-to-Maturity)
                "r": rf,           # 无风险利率 (Short Rate)
                "sigma": volatility,       # 波动率 (Volatility)
                "q": q, # 股息收益率
                
                # --- 模拟参数 ---
                "I": N_inner,           # 模拟路径数量
                "M": 100,           # 时间步数量
            }
            V1_T = American_MC_calculation(params)
        else:
            print(f"style必须是European和American中的一个!")

        #计算T时刻外层期权的收益
        payoff_outer_T = np.maximum(N1*(V1_T - K1),0)
        #计算0时刻外层期权的价格
        discounted_payoffs[j] = discount_T * payoff_outer_T

    #分析内层期权T时刻的价格，以正确模拟K1
    # V1_Ts = np.array(V1_Ts)
    # V1_T_mean = np.mean(V1_Ts)
    # V1_T_median = np.median(V1_Ts)

    # print(f"内层期权在T时刻的价格平均值为:{V1_T_mean},价格中位数为{V1_T_median}")

    price_X = discounted_payoffs.mean()
    #print(f"外层{N_outer}步内层{N_inner}步的蒙特卡洛定价法得到期权X的价格为:{price_X}")
    return price_X


print(f"内层为欧式期权的X的蒙特卡洛模拟定价为:{monte_carlo_priceX('European')}")
print(f"内层为美式期权的X的蒙特卡洛模拟定价为:{monte_carlo_priceX('American')}")
import math
import pandas as pd
import numpy as np

np.random.seed(150000)

# 定义参数模板
MODEL_PARAMS_TEMPLATE = {
    # --- 模型参数 ---
    "S0": None,          # 初始股价 (Initial Stock Level)
    "K": None,           # 行权价格 (Strike Price)
    "T": None,           # 到期时间 (Time-to-Maturity)
    "r": None,           # 无风险利率 (Short Rate)
    "sigma": None,       # 波动率 (Volatility)
    "q": 0.0, # 股息收益率
    
    # --- 模拟参数 ---
    "I": None,           # 模拟路径数量
    "M": None,           # 时间步数量
}

def American_MC_calculation(params_dict):
    #获取参数
    try:
        S0 = params_dict["S0"]
        K = params_dict["K"]
        T = params_dict["T"]
        r = params_dict["r"]
        sigma = params_dict["sigma"]
        q = params_dict["q"]  # 股息收益率
        I = params_dict["I"]
        M = params_dict["M"]

    except KeyError as e:
        raise ValueError(f"参数字典中缺少关键参数: {e}. 请检查您的字典构建是否完整。")
    dt = T/M
    df = np.exp(-r*dt)

    S = S0 * np.exp(np.cumsum((r-q - 0.5 * sigma ** 2) * dt + \
                            sigma * math.sqrt(dt) * np.random.standard_normal((M + 1, I)), axis=0))
    S[0] = S0
    h = np.maximum(K - S**1.5, 0)
    V = h[-1]

    for t in range(M - 1, 0, -1):
        rg = np.polyfit(S[t], V * df, 5)#使用5次多项式回归
        C = np.polyval(rg, S[t]) 
        V = np.where(h[t] > C, h[t], V * df)
    
    V0 = df * np.sum(V) / I # LSM estimator

    #print(f"American put option value{V0:.3f}") 
    return V0

def European_MC_calculation(params_dict):
    #获取参数
    try:
        S0 = params_dict["S0"]
        K = params_dict["K"]
        T = params_dict["T"]
        r = params_dict["r"]
        sigma = params_dict["sigma"]
        q = params_dict["q"]  # 股息收益率
        I = params_dict["I"]
        
    except KeyError as e:
        raise ValueError(f"参数字典中缺少关键参数: {e}. 请检查您的字典构建是否完整。")
    mu = r-q - 0.5 * sigma**2
    sig = sigma*np.sqrt(T)
    discount_T = np.exp(-r*T)
    S = S0*np.exp(mu*T + sig*np.random.standard_normal(I))
    h = np.maximum(K - S**1.5, 0)
    V0 = h.mean()*discount_T
    return V0


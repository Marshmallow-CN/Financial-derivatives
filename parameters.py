import pandas as pd
import numpy as np

# Parameters for binomial tree model

rf = 0.0428  # annual risk-free interest rate 年化无风险收益率
volatility = 0.3447  # annual volatility of the underlying asset 年化波动率
q = 0.008  # annaul dividend yield 年化股息收益率
USD_HKD = 0.13 # exchange rate HKD to USD / 1港元兑换多少美元

#===内层期权参数===
S0 = 81.31 # initial stock price of Tencent, dollar , 2025-11-05
K2 = [580,590,600,610,620,630,640,650,660,670,680]  # strike prices in HKD 内层期权执行价 港元
K2 = [((k * USD_HKD) ** 1.5) for k in K2]  # strike prices in USD 内层期权执行价 美元
T_plus_C = pd.to_datetime("2026-05-26") #T_plus_C: 内层期权到期日
T = pd.to_datetime("2026-2-05") #T: 外层期权到期日/内层期权开始日
t = pd.to_datetime("2025-11-05") # t为当前日期 
delta_inner = T_plus_C - T
delta_outer = T - t
delta_all = T_plus_C - t
print("外层期权到期时间天数:", delta_outer.days)
print("内层期权到期时间天数:", delta_inner.days)
C = delta_all.days/365  # time to maturity in years, 3 months , 2026-02-26号到期 内层期权到期时间 年
C_inner = delta_inner.days/365  
C_outer = delta_outer.days/365  
print("内层期权到期时间(年):", C_inner, "外层期权到期时间(年):", C_outer, "内外层期权总时间(年):", C)
#===内层期权参数===

#===外层期权参数===
K1 = 1
N1 = 1
N2 = -1
#===外层期权参数===


#===二项式模型参数===
M = 100 # number of time steps 步数
dt = C / M  # length of time step 每步时间
M_inner = int(M * (C_inner / C))  # 内层期权步数
M_outer = M - M_inner  # 外层期权步数
print("内层期权步数:", M_inner, "外层期权步数:", M_outer)
df = np.exp(-rf * dt)  # discount factor per time step 每步贴现因子
u = np.exp(volatility * np.sqrt(dt))  # up-factor per time step 上行因子
d = 1 / u  # down-factor per time step 下行因子
p = (np.exp((rf - q) * dt) - d) / (u - d)  # risk-neutral up probability 风险中性概率
#===二项式模型参数===


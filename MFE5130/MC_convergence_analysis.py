import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from parameters import *

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 方案2：向量化 + 渐进式采样（超快速版本）
# 运行时间：约1-2分钟（相比原来的几小时）

def Monte_Carlo_convergence_fast(out_number, max_in_number, N_inner_values, seed=0):
    """
    快速计算不同N_inner下的蒙特卡洛价格
    关键优化：一次性生成所有随机数，然后渐进式采样
    """
    rng = np.random.default_rng(seed)
    
    # 一次性生成外层随机数
    Z = rng.standard_normal(out_number)
    muT = rf - q - 0.5 * volatility**2 
    sigT = volatility*np.sqrt(C_outer)
    S_T = S0*np.exp(muT*C_outer + sigT*Z)
    
    muC = rf - q - 0.5 * volatility**2 
    sigC = volatility*np.sqrt(C_inner)
    discount_C = np.exp(-rf*C_inner)
    discount_T = np.exp(-rf*C_outer)
    
    # 一次性生成所有内层随机数 (out_number, max_in_number)
    print(f"一次性生成 {out_number} x {max_in_number} = {out_number * max_in_number:,} 个随机数...")
    Z_inner_all = rng.standard_normal((out_number, max_in_number))
    
    # 预计算所有股价路径
    S_T_expanded = S_T[:, np.newaxis]
    S_TC_all = S_T_expanded * np.exp(muC*C_inner + sigC * Z_inner_all)
    
    # 预计算所有payoff
    payoff_inner_all = np.maximum(K2[5] - S_TC_all ** 1.5, 0)
    
    # 对于每个N_inner值，只需要取前N_inner列求平均
    prices = []
    print(f"计算 {len(N_inner_values)} 个不同N_inner的价格...")
    
    for idx, n_inner in enumerate(N_inner_values):
        V1_T = payoff_inner_all[:, :n_inner].mean(axis=1) * discount_C
        payoff_outer_T = np.maximum(N1*(V1_T - K1), 0)
        price = (discount_T * payoff_outer_T).mean()
        prices.append(price)
        
        if (idx + 1) % 10 == 0:
            print(f"进度: {idx+1}/{len(N_inner_values)}, N_inner={n_inner}, 价格={price:.4f}")
    
    return prices

# 设置采样点
N_inner_values = np.unique(np.logspace(0, 3, 50, dtype=int))
print(f"采样点数量: {len(N_inner_values)}")
print(f"采样点范围: {N_inner_values[0]} 到 {N_inner_values[-1]}\n")

# 运行快速计算
import time
start_time = time.time()
Monte_Carlo_output = Monte_Carlo_convergence_fast(
    out_number=10000, 
    max_in_number=1000, 
    N_inner_values=N_inner_values,
    seed=0
)
elapsed_time = time.time() - start_time

print(f"\n✓ 计算完成！")
print(f"  运行时间: {elapsed_time:.2f} 秒")
print(f"  最终价格 (N_inner=1000): {Monte_Carlo_output[-1]:.4f}")
print(f"  二叉树价格: 48.43")
print(f"  差异: {abs(Monte_Carlo_output[-1] - 48.43):.4f}\n")

# 转换为numpy数组以便索引
Monte_Carlo_output = np.array(Monte_Carlo_output)

# 创建对比图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 左图：对数坐标
ax1.plot(N_inner_values, Monte_Carlo_output, 'b-o', markersize=5, 
         label='Monte Carlo Pricing', linewidth=2, alpha=0.7)
ax1.axhline(y=48.43, color='r', linestyle='--', linewidth=2, label='Binomial Tree (48.43)')
ax1.fill_between(N_inner_values, 48.43-0.5, 48.43+0.5, alpha=0.2, color='red', 
                  label='±0.5 Range')
ax1.set_xlabel('Inner Simulation Steps (N_inner)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Option Price ($)', fontsize=12, fontweight='bold')
ax1.set_title('Monte Carlo Convergence Analysis (Log Scale)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')

# 右图：线性坐标（聚焦后半段）
mask = N_inner_values >= 100
ax2.plot(N_inner_values[mask], Monte_Carlo_output[mask], 'b-o', markersize=5, 
         label='Monte Carlo Pricing', linewidth=2, alpha=0.7)
ax2.axhline(y=48.43, color='r', linestyle='--', linewidth=2, label='Binomial Tree (48.43)')
ax2.fill_between([100, 1000], 48.43-0.5, 48.43+0.5, alpha=0.2, color='red')
ax2.set_xlabel('Inner Simulation Steps (N_inner)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Option Price ($)', fontsize=12, fontweight='bold')
ax2.set_title('Convergence Detail (N_inner ≥ 100)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10, loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_xlim([100, 1000])

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/MC_vs_Binomial_Convergence.png', dpi=300, bbox_inches='tight')
print("图像已保存: MC_vs_Binomial_Convergence.png")

# 保存详细数据
result_df = pd.DataFrame({
    'N_inner': N_inner_values,
    'MC_Price': Monte_Carlo_output,
    'Binomial_Price': 48.43,
    'Difference': np.array(Monte_Carlo_output) - 48.43,
    'Abs_Difference': np.abs(np.array(Monte_Carlo_output) - 48.43)
})
result_df.to_csv('/mnt/user-data/outputs/MC_convergence_data.csv', index=False)
print("数据已保存: MC_convergence_data.csv")

# 打印统计信息
print("\n=== 收敛性统计 ===")
print(f"N_inner=100时的价格: {result_df[result_df['N_inner']>=100]['MC_Price'].iloc[0]:.4f}")
print(f"N_inner=500时的价格: {result_df[result_df['N_inner']>=500]['MC_Price'].iloc[0]:.4f}")
print(f"N_inner=1000时的价格: {Monte_Carlo_output[-1]:.4f}")
print(f"平均绝对误差 (N_inner≥100): {result_df[result_df['N_inner']>=100]['Abs_Difference'].mean():.4f}")

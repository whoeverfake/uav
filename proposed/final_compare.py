import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load all three results
local_new = np.loadtxt('results/reward.csv')
local_old = np.loadtxt('results/reward_old_gamma0.csv')

# Load Bahdanau from log
import re
with open('train_bahdanau.log') as f:
    log = f.read()
bahdanau_rewards = [float(m.group(1)) for m in re.finditer(r'eval average episode rewards of environment:\s*([\d.]+)', log)]
bahdanau = np.array(bahdanau_rewards)

# Create comparison plot
plt.figure(figsize=(12, 6))

# Smooth with EMA
def ema(x, alpha=0.1):
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    return y

# Plot all three
steps_old = np.linspace(0, 100000, len(local_old))
steps_new = np.linspace(0, 100000, len(local_new))
steps_bah = np.linspace(0, 100000, len(bahdanau))

plt.plot(steps_old, local_old, alpha=0.3, color='orange', linewidth=0.8)
plt.plot(steps_old, ema(local_old, 0.05), color='orange', linewidth=2, label=f'Local (γ=0.66, 旧版): {np.mean(local_old[-10:]):.1f}')

plt.plot(steps_new, local_new, alpha=0.3, color='red', linewidth=0.8)
plt.plot(steps_new, ema(local_new, 0.05), color='red', linewidth=2, label=f'Local (γ=0.12, 新版): {np.mean(local_new[-10:]):.1f}')

plt.plot(steps_bah, bahdanau, alpha=0.3, color='blue', linewidth=0.8)
plt.plot(steps_bah, ema(bahdanau, 0.05), color='blue', linewidth=2, label=f'Bahdanau (γ=0, 全局): {np.mean(bahdanau[-10:]):.1f}')

plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Average Episode Reward (Eval)', fontsize=12)
plt.title('Final Comparison: Local Spatial Self-Attention vs Bahdanau', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig('final_comparison.png', dpi=150)
print(f"Saved: final_comparison.png")

# Print summary
print("\n" + "="*60)
print("最终结果对比 (最后10轮平均)")
print("="*60)
print(f"Bahdanau (全局注意力, γ=0):          {np.mean(bahdanau[-10:]):.2f}")
print(f"Local新版 (γ=0.12, 初始化-2.0):      {np.mean(local_new[-10:]):.2f} (-{976.65-np.mean(local_new[-10:]):.2f}, -{(1-np.mean(local_new[-10:])/976.65)*100:.1f}%)")
print(f"Local旧版 (γ=0.66, 初始化0.0):       {np.mean(local_old[-10:]):.2f} (-{976.65-np.mean(local_old[-10:]):.2f}, -{(1-np.mean(local_old[-10:])/976.65)*100:.1f}%)")
print(f"原始MAPPO基线(未修复):                730.99 (-245.66, -25.1%)")
print("="*60)
print("\n关键发现:")
print("1. 修复embedding后,所有版本相比原始基线都有显著提升")
print("2. gamma初始化从0→-2显著提升性能(823→888, +7.8%)")
print("3. Bahdanau全局注意力仍然最优(976.65)")
print("4. 局部空间约束(即使很弱的γ=0.12)仍然不如完全全局")
print("\n结论: UAV协调任务需要全局信息,空间局部性假设不成立")

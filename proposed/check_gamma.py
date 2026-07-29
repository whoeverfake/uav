import torch
import torch.nn.functional as F

# Load critic model
critic = torch.load('results/models/critic.pt', map_location='cpu')

# Check if log_gamma exists
if 'attention.log_gamma' in critic:
    log_gamma = critic['attention.log_gamma'].item()
    gamma = F.softplus(torch.tensor(log_gamma)).item()
    
    print(f"Trained log_gamma: {log_gamma:.6f}")
    print(f"Softplus(log_gamma) = gamma: {gamma:.6f}")
    print(f"\nInterpretation:")
    print(f"  - gamma=0: 完全全局,无距离惩罚")
    print(f"  - gamma=1: 中等局部性,距离200时衰减e^(-1)≈0.37")
    print(f"  - gamma>5: 强局部性,远距离几乎不可见")
    print(f"\n当前gamma={gamma:.2f},在radius=200下:")
    import math
    for d in [50, 100, 200, 300, 400]:
        bias = -gamma * (d/200.0)**2
        weight = math.exp(bias)
        print(f"  距离{d}m: 注意力权重衰减到 {weight:.4f}")
else:
    print("log_gamma not found - this model uses Bahdanau attention")

print("\n\n检查其他关键参数:")
for key in sorted(critic.keys()):
    if 'attention' in key:
        val = critic[key]
        if val.numel() == 1:
            print(f"{key}: {val.item():.6f}")
        else:
            print(f"{key}: shape {list(val.shape)}, mean={val.mean().item():.4f}, std={val.std().item():.4f}")

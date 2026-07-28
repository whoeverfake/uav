"""Smoke-test for LocalSpatialSelfAttention and the attention-based R_Critic."""
import sys, os, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_result.log")
_lines = []


def print(*args, **kwargs):  # noqa: A001 - tee stdout into a log file
    msg = " ".join(str(a) for a in args)
    _lines.append(msg)
    sys.stdout.write(msg + "\n")


def _flush():
    with open(_LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")


try:
    import torch
    from algorithms.utils.attention import LocalSpatialSelfAttention
except Exception:
    _lines.append("IMPORT FAILED:\n" + traceback.format_exc())
    _flush()
    raise


def make_obs(B, M, D=4, spread=500.0):
    """Build a realistic flattened cent_obs: [x, y, n_users, throughput] per UAV."""
    x = torch.rand(B, M, D)
    x[:, :, :2] *= spread          # coords in [0, spread]
    x[:, :, 2] *= 200              # user counts
    x[:, :, 3] *= 30               # throughput
    return x.view(B, M * D)


# ── 1. shape correctness across agent counts ─────────────────────────────────
for M in (3, 4, 5, 6, 8):
    attn = LocalSpatialSelfAttention(M, 4, 32, n_heads=1, radius=200.0)
    out = attn(make_obs(4, M))
    assert out.shape == (4, 32), f"M={M}: bad shape {out.shape}"
print("[PASS] output shape correct for M in {3,4,5,6,8}")

# ── 2. multi-head ────────────────────────────────────────────────────────────
attn4 = LocalSpatialSelfAttention(6, 4, 32, n_heads=4, radius=200.0)
assert attn4(make_obs(4, 6)).shape == (4, 32)
print("[PASS] 4-head attention works")

# ── 3. no NaN even when every agent is isolated (tiny radius) ────────────────
attn_tiny = LocalSpatialSelfAttention(6, 4, 32, radius=1e-6)
out_tiny = attn_tiny(make_obs(8, 6))
assert torch.isfinite(out_tiny).all(), "NaN/Inf with isolated agents"
print("[PASS] isolated agents (radius->0) produce no NaN")

# ── 4. locality actually changes the result ─────────────────────────────────
torch.manual_seed(0)
small = LocalSpatialSelfAttention(6, 4, 32, radius=50.0)
torch.manual_seed(0)
large = LocalSpatialSelfAttention(6, 4, 32, radius=10_000.0)
obs = make_obs(4, 6)
assert not torch.allclose(small(obs), large(obs)), \
    "radius has no effect - spatial mask may not be applied"
print("[PASS] spatial radius changes the output (mask is active)")

# ── 5. R_Critic end to end ──────────────────────────────────────────────────
import utils.util as uutil
uutil.get_shape_from_obs_space = lambda s: s.shape     # lightweight stub

from algorithms.algorithm.r_actor_critic import R_Critic

class Args:
    hidden_size = 32
    use_orthogonal = True
    use_popart = False
    obs_per_agent = 4
    attn_radius = 200.0
    attn_heads = 1

class Space:
    def __init__(self, dim): self.shape = (dim,)

B, M = 4, 6
critic = R_Critic(Args(), Space(M * 4))
assert critic.attention.n_agents == M, "agent count not inferred correctly"
obs = make_obs(B, M)
values, _ = critic(obs, torch.zeros(B, 1, 32), torch.ones(B, 1))
assert values.shape == (B, 1), f"bad values shape {values.shape}"
print(f"[PASS] R_Critic inferred M={M}, values shape {tuple(values.shape)}")

# ── 6. gradient flow ────────────────────────────────────────────────────────
values.mean().backward()
missing = [n for n, p in critic.named_parameters() if p.requires_grad and p.grad is None]
assert not missing, f"no gradient for: {missing}"
print("[PASS] gradients reach all critic parameters")

# ── 7. param count vs the old MLP+GRU critic ────────────────────────────────
n_params = sum(p.numel() for p in critic.parameters())
print(f"attention critic parameters: {n_params:,}")
print("All tests passed.")
_flush()

"""
VIPER RL 
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd

torch.manual_seed(42)
np.random.seed(42)

print("Loading stock data...")
data = pd.read_csv(".gitignore/all_stocks.csv")
data = data[['date', 'Name', 'close']].sort_values('date')
price_matrix = data.pivot(index='date', columns='Name', values='close').dropna(axis=1)
returns_np = np.log(price_matrix / price_matrix.shift(1)).dropna()
returns_raw = torch.tensor(returns_np.values[:, :10], dtype=torch.float32)

T, N = returns_raw.shape
print(f"✓ Loaded {T} timesteps, {N} assets\n")

# Get 20-day rolling stats as state
window = 20

def get_state(returns, t):
    """Get mean/std/last from past window"""
    if t < window:
        return torch.zeros(3 * N)
    w = returns[t - window:t]
    mean = w.mean(dim=0)
    std = w.std(dim=0) + 1e-8
    last = w[-1]
    return torch.cat([mean, std, last])

states_raw = torch.stack([get_state(returns_raw, t) for t in range(window, T)])
print(f"State shape: {states_raw.shape}\n")

# VAE for state compression
class VAE(nn.Module):
    def __init__(self, input_dim=30, latent_dim=5):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(8, latent_dim)
        self.fc_logvar = nn.Linear(8, latent_dim)
        
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 8), nn.ReLU(),
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, input_dim),
        )
    
    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.dec(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Train VAE
vae = VAE(input_dim=30, latent_dim=5)
vae_opt = optim.Adam(vae.parameters(), lr=0.001)

for epoch in range(10):
    vae_opt.zero_grad()
    recon, mu, logvar = vae(states_raw)
    
    mse = F.mse_loss(recon, states_raw)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = mse + 0.001 * kl
    
    loss.backward()
    vae_opt.step()

print("✓ VAE trained\n")

# Actor and Critic
class Actor(nn.Module):
    def __init__(self, latent_dim, n_assets):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, n_assets),
        )
    
    def forward(self, z):
        logits = self.net(z)
        return torch.softmax(logits, dim=-1)

class Critic(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )
    
    def forward(self, z):
        return self.net(z)

# Eval function - log returns for compound calculations, convert to simple for Sharpe
def eval_portfolio(weights, returns, steps=1258):
    daily_ret = []
    for t in range(min(steps, returns.shape[0])):
        daily_ret.append(torch.dot(weights, returns[t]).item())
    
    daily_ret = np.array(daily_ret)
    n = len(daily_ret)
    
    # Annualized return from log returns
    total_log_ret = daily_ret.sum()
    ann_ret = (np.exp(total_log_ret * (252 / n)) - 1) * 100 if n > 0 else 0
    
    # Convert log returns to simple for proper risk metrics
    simple_ret = np.exp(daily_ret) - 1
    
    # Sharpe from simple returns
    sharpe = (simple_ret.mean() / max(simple_ret.std(), 1e-4)) * np.sqrt(252)
    
    # Max drawdown in % wealth loss terms 
    cs = np.cumsum(daily_ret)
    peak = np.maximum.accumulate(cs)
    max_dd_log = (cs - peak).min() if len(cs) > 0 else 0
    max_dd = (1 - np.exp(max_dd_log)) * 100  # Convert to % loss
    
    # Calmar - return % / drawdown %
    calmar = ann_ret / max(max_dd, 1e-8)
    
    # Sortino - downside risk from simple returns
    downside = simple_ret[simple_ret < 0]
    downside_std = np.std(downside) if len(downside) > 0 else simple_ret.std()
    sortino = (simple_ret.mean() / max(downside_std, 1e-4)) * np.sqrt(252)
    
    # Win rate
    win_rate = (simple_ret > 0).sum() / len(simple_ret) * 100 if len(simple_ret) > 0 else 0
    
    return ann_ret, sharpe, max_dd, calmar, sortino, win_rate

print("="*75)
print("VIPER RL: ADVERSARIAL PORTFOLIO ROBUSTNESS")
print("="*75)

# Baseline - equal weight
print("\n[1] BASELINE (Equal Weight)")
print("-" * 75)

w_baseline = torch.ones(N) / N
base_ret, base_sharpe, base_mdd, base_calmar, base_sortino, base_wr = eval_portfolio(w_baseline, returns_raw, steps=1258)

print(f"Return:     {base_ret:7.2f}%")
print(f"Sharpe:     {base_sharpe:7.3f}")
print(f"Max DD:     {base_mdd:8.6f}")
print(f"Calmar:     {base_calmar:7.3f}")
print(f"Sortino:    {base_sortino:7.3f}")
print(f"Win Rate:   {base_wr:6.1f}%\n")

# Train a policy
print("[2] TRAINING CLEAN POLICY (50 steps)")
print("-" * 75)

actor_weights = nn.Parameter(torch.zeros(N))
w_opt = optim.Adam([actor_weights], lr=0.01)

for step in range(50):
    w_opt.zero_grad()
    
    w = torch.softmax(actor_weights, dim=0)
    
    # just maximize returns on a window
    ret_sum = 0
    for t in range(150):
        ret_sum = ret_sum + torch.dot(w, returns_raw[t])
    
    loss = -ret_sum
    loss.backward()
    w_opt.step()

w_clean_avg = torch.softmax(actor_weights.detach(), dim=0)

clean_ret, clean_sharpe, clean_mdd, clean_calmar, clean_sortino, clean_wr = eval_portfolio(w_clean_avg, returns_raw, steps=1258)

print(f"Return:     {clean_ret:7.2f}%")
print(f"Sharpe:     {clean_sharpe:7.3f}")
print(f"Max DD:     {clean_mdd:8.6f}")
print(f"Calmar:     {clean_calmar:7.3f}")
print(f"Sortino:    {clean_sortino:7.3f}")
print(f"Win Rate:   {clean_wr:6.1f}%")
print(f"vs Baseline: +{clean_ret - base_ret:.2f}%\n")

# Attack baseline
print("[3] ATTACKING BASELINE")
print("-" * 75)

delta_base = nn.Parameter(torch.randn(N) * 0.0015)
atk_opt = optim.Adam([delta_base], lr=0.005)

for step in range(20):
    atk_opt.zero_grad()
    
    w_pert = w_baseline + delta_base
    w_pert = torch.clamp(w_pert, min=0)
    w_pert = w_pert / (w_pert.sum() + 1e-8)
    
    ret_sum = 0
    for t in range(150):
        ret_sum = ret_sum + torch.dot(w_pert, returns_raw[t])
    
    atk_loss = ret_sum + 0.01 * torch.norm(delta_base)
    atk_loss.backward()
    atk_opt.step()
    delta_base.data.clamp_(-0.12, 0.12)

w_base_attacked = w_baseline + delta_base.detach()
w_base_attacked = torch.clamp(w_base_attacked, min=0)
w_base_attacked = w_base_attacked / (w_base_attacked.sum() + 1e-8)

print(f"Original weights: {w_baseline.tolist()[:3]}... (equal)")
print(f"Attacked weights: {w_base_attacked.tolist()[:3]}...")
print(f"Weight shift:     {delta_base.norm().item():.6f}\n")

base_att_ret, base_att_sharpe, base_att_mdd, base_att_calmar, base_att_sortino, base_att_wr = eval_portfolio(w_base_attacked, returns_raw, steps=1258)
base_attack_impact = base_ret - base_att_ret

print(f"Return:     {base_att_ret:7.2f}%")
print(f"Sharpe:     {base_att_sharpe:7.3f}")
print(f"Max DD:     {base_att_mdd:8.6f}")
print(f"Calmar:     {base_att_calmar:7.3f}")
print(f"Sortino:    {base_att_sortino:7.3f}")
print(f"Win Rate:   {base_att_wr:6.1f}%")
if base_attack_impact > 0:
    print(f"✗ Returns REDUCED by {base_attack_impact:.2f}pp (attack worked)")
else:
    print(f"✓ Returns INCREASED by {-base_attack_impact:.2f}pp (attack failed)")
print()

# Attack clean policy
print("[4] ATTACKING CLEAN POLICY")
print("-" * 75)

delta_clean = nn.Parameter(torch.randn(N) * 0.0015)
atk_opt2 = optim.Adam([delta_clean], lr=0.005)

for step in range(20):
    atk_opt2.zero_grad()
    
    w_pert = w_clean_avg + delta_clean
    w_pert = torch.clamp(w_pert, min=0)
    w_pert = w_pert / (w_pert.sum() + 1e-8)
    
    ret_sum = 0
    for t in range(150):
        ret_sum = ret_sum + torch.dot(w_pert, returns_raw[t])
    
    atk_loss = ret_sum + 0.01 * torch.norm(delta_clean)
    atk_loss.backward()
    atk_opt2.step()
    delta_clean.data.clamp_(-0.12, 0.12)

w_clean_attacked = w_clean_avg + delta_clean.detach()
w_clean_attacked = torch.clamp(w_clean_attacked, min=0)
w_clean_attacked = w_clean_attacked / (w_clean_attacked.sum() + 1e-8)

print(f"Original weights: {w_clean_avg.tolist()[:3]}...")
print(f"Attacked weights: {w_clean_attacked.tolist()[:3]}...")
print(f"Weight shift:     {delta_clean.norm().item():.6f}\n")

clean_att_ret, clean_att_sharpe, clean_att_mdd, clean_att_calmar, clean_att_sortino, clean_att_wr = eval_portfolio(w_clean_attacked, returns_raw, steps=1258)
clean_attack_impact = clean_ret - clean_att_ret

print(f"Return:     {clean_att_ret:7.2f}%")
print(f"Sharpe:     {clean_att_sharpe:7.3f}")
print(f"Max DD:     {clean_att_mdd:8.6f}")
print(f"Calmar:     {clean_att_calmar:7.3f}")
print(f"Sortino:    {clean_att_sortino:7.3f}")
print(f"Win Rate:   {clean_att_wr:6.1f}%")
if clean_attack_impact > 0:
    print(f"✗ Returns REDUCED by {clean_attack_impact:.2f}pp (attack worked)")
else:
    print(f"✓ Returns INCREASED by {-clean_attack_impact:.2f}pp (attack failed)")
print()

# Summary
print("="*75)
print("SUMMARY")
print("="*75)

print(f"\nBaseline:            {base_ret:7.2f}% (Sharpe {base_sharpe:.3f})")
print(f"  └─ Under Attack:   {base_att_ret:7.2f}% (lost {base_attack_impact:.2f}pp)")

print(f"\nClean Policy:        {clean_ret:7.2f}% (Sharpe {clean_sharpe:.3f})")
print(f"  └─ Under Attack:   {clean_att_ret:7.2f}% (lost {clean_attack_impact:.2f}pp)")

print(f"\nPolicy improvement: +{clean_ret - base_ret:.2f}%")
print(f"Robustness gain:    {base_attack_impact - clean_attack_impact:.2f}pp")

if clean_ret > base_ret and clean_attack_impact < base_attack_impact:
    print("\n✓ Policy is better AND more robust!")
elif clean_ret > base_ret:
    print("\n✓ Policy is better (but not more robust to attacks)")
elif clean_attack_impact < base_attack_impact:
    print("\n✓ Policy is more robust (but lower returns)")
else:
    print("\n✗ Policy is worse")

print("="*75)

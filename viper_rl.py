"""
VIPER RL - Stackelberg Game for Portfolio Robustness
Defender: Actor learns robust portfolio weights from latent states
Attacker: Uses optimization to find adversarial perturbations in latent space
Game Loop: Alternating optimization (similar to MNIST CNN-VAE)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from copy import deepcopy

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("VIPER RL - Stackelberg Game for Portfolio Robustness")
print("-" * 80)

# ============================================================================
# [1] LOAD DATA
# ============================================================================
print("\n[1] Loading data...")

data = pd.read_csv(".gitignore/all_stocks.csv")
data = data[['date', 'Name', 'close']].sort_values('date')
price_matrix = data.pivot(index='date', columns='Name', values='close').dropna(axis=1)

returns_np = np.log(price_matrix / price_matrix.shift(1)).dropna()
returns_data = torch.tensor(returns_np.values[:, :10], dtype=torch.float32).to(device)

n_days, n_assets = returns_data.shape
print(f"Data: {n_days} days, {n_assets} assets")

# ============================================================================
# [2] CREATE STATE REPRESENTATION
# ============================================================================
print("\n[2] Creating states...")

WINDOW = 10

def create_state(returns, t, window, n_assets):
    if t < window:
        return torch.zeros(4 * n_assets, device=device)
    
    w_returns = returns[t - window:t]
    means = w_returns.mean(dim=0)
    stds = w_returns.std(dim=0) + 1e-8
    mins = w_returns.min(dim=0)[0]
    maxs = w_returns.max(dim=0)[0]
    
    state = torch.cat([means, stds, mins, maxs])
    return state

states = torch.stack([create_state(returns_data, t, WINDOW, n_assets) for t in range(WINDOW, n_days)])
state_dim = states.shape[1]
print(f"States shape: {states.shape}")

# ============================================================================
# [3] VAE - ENCODE STATES TO LATENT
# ============================================================================
print("\n[3] Building VAE...")

LATENT_DIM = 10

class VAE(nn.Module):
    def __init__(self, state_dim, latent_dim):
        super().__init__()
        
        # Encoder
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 32)
        self.fc4 = nn.Linear(32, 64)
        self.fc5 = nn.Linear(64, state_dim)
    
    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        x_recon = self.fc5(h)
        return x_recon
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return z, mu, logvar, x_recon

vae = VAE(state_dim, LATENT_DIM).to(device)
vae_opt = optim.Adam(vae.parameters(), lr=0.001)

# Train VAE
print("Training VAE...")
for epoch in range(20):
    vae_opt.zero_grad()
    z, mu, logvar, x_recon = vae(states)
    
    mse_loss = F.mse_loss(x_recon, states)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = mse_loss + 0.001 * kl_loss
    
    loss.backward()
    vae_opt.step()

print("VAE trained")

# Get latent states
vae.eval()
with torch.no_grad():
    latent_states, _, _, _ = vae(states)
print(f"Latent states: {latent_states.shape}")

# ============================================================================
# [4] ACTOR-CRITIC - DEFENDER
# ============================================================================
print("\n[4] Building Actor-Critic (Defender)...")

class Actor(nn.Module):
    def __init__(self, latent_dim, n_assets):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, n_assets),
        )
    
    def forward(self, z):
        logits = self.net(z)
        weights = torch.softmax(logits, dim=-1)
        return weights

class Critic(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    
    def forward(self, z):
        value = self.net(z)
        return value

actor = Actor(LATENT_DIM, n_assets).to(device)
critic = Critic(LATENT_DIM).to(device)

actor_opt = optim.Adam(actor.parameters(), lr=0.001)
critic_opt = optim.Adam(critic.parameters(), lr=0.001)
print("Actor-Critic ready")

# ============================================================================
# [5] METRICS
# ============================================================================
def compute_metrics(weights, returns):
    """Return, Sharpe, Sortino, Max DD, Win Rate"""
    daily_log_ret = (weights * returns).sum(dim=1)
    n = len(daily_log_ret)
    
    # Return
    total_log = daily_log_ret.sum().item()
    annual_ret = (np.exp(total_log * 252 / n) - 1) * 100
    
    # Sharpe
    simple_ret = torch.exp(daily_log_ret) - 1
    mean_ret = simple_ret.mean().item()
    std_ret = simple_ret.std().item()
    sharpe = (mean_ret / (std_ret + 1e-8)) * np.sqrt(252)
    
    # Sortino
    downside = simple_ret[simple_ret < 0]
    downside_std = downside.std().item() if len(downside) > 0 else std_ret
    sortino = (mean_ret / (downside_std + 1e-8)) * np.sqrt(252)
    
    # Max DD
    cum_log = torch.cumsum(daily_log_ret, dim=0)
    running_max = torch.cummax(cum_log, dim=0)[0]
    max_dd = (1 - torch.exp((cum_log - running_max).min()).item()) * 100
    
    # Win rate
    win_rate = (simple_ret > 0).sum().item() / len(simple_ret) * 100
    
    return {
        'return': annual_ret,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd': max_dd,
        'win_rate': win_rate,
    }

# ============================================================================
# [6] ATTACKER - FIND ADVERSARIAL PERTURBATIONS
# ============================================================================
def attacker_optimize(actor_model, latent_states_subset, returns_subset, n_iter=50):
    """Find perturbations that minimize portfolio return using gradient ascent"""
    actor_model.eval()
    
    # Initialize perturbation
    delta = torch.zeros_like(latent_states_subset, requires_grad=True)
    delta_opt = optim.Adam([delta], lr=0.01)
    
    best_loss = float('inf')
    best_delta = delta.data.clone()
    
    for i in range(n_iter):
        delta_opt.zero_grad()
        
        # Perturbed latent states
        z_perturbed = latent_states_subset + delta
        
        # Get weights
        weights = actor_model(z_perturbed)
        
        # Compute returns (negative because we want to minimize)
        daily_log_ret = (weights * returns_subset).sum(dim=1)
        loss = daily_log_ret.mean()  # Minimize return
        
        (-loss).backward()
        delta_opt.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_delta = delta.data.clone()
    
    return best_delta.detach()

# ============================================================================
# [7] MAIN GAME LOOP
# ============================================================================
print("\n[5] Game Loop...")

REBAL_FREQ = 15
TC = 0.001
PENALTY = 0.1
NUM_EPISODES = 5

for episode in range(NUM_EPISODES):
    print(f"\nEpisode {episode + 1}:")
    
    # Defender plays: collect trajectory with clean actor
    print("  Defender: playing on clean data...")
    actor.eval()
    weights_clean = []
    returns_clean = []
    
    with torch.no_grad():
        for t in range(len(latent_states) - REBAL_FREQ):
            z_t = latent_states[t]
            w_t = actor(z_t.unsqueeze(0)).squeeze()
            weights_clean.append(w_t)
            returns_clean.append(returns_data[WINDOW + t])
    
    weights_clean = torch.stack(weights_clean)
    returns_clean = torch.stack(returns_clean)
    clean_metrics = compute_metrics(weights_clean, returns_clean)
    print(f"    Clean return: {clean_metrics['return']:.2f}%")
    
    # Attacker plays: find perturbations that hurt returns
    print("  Attacker: finding adversarial perturbations...")
    delta_attack = attacker_optimize(actor, latent_states[:-REBAL_FREQ], returns_data[WINDOW:WINDOW + len(latent_states) - REBAL_FREQ], n_iter=10)
    
    # Attack: apply perturbations and get weights
    with torch.no_grad():
        z_attacked = latent_states[:-REBAL_FREQ] + delta_attack
        weights_attacked = actor(z_attacked)
        returns_attacked = returns_data[WINDOW:WINDOW + len(latent_states) - REBAL_FREQ]
    
    attacked_metrics = compute_metrics(weights_attacked, returns_attacked)
    print(f"    Attacked return: {attacked_metrics['return']:.2f}%")
    print(f"    Attack loss: {attacked_metrics['return'] - clean_metrics['return']:.2f}pp")
    
    # Defender adapts: train on adversarial data
    print("  Defender: training actor-critic on adversarial data...")
    actor.train()
    critic.train()
    
    for epoch in range(10):
        actor_opt.zero_grad()
        critic_opt.zero_grad()
        
        weights_adv = actor(z_attacked)
        values = critic(z_attacked).squeeze()
        
        # Compute return
        daily_ret = (weights_adv * returns_attacked).sum(dim=1)
        
        # Advantage
        advantage = daily_ret - values.detach()
        
        # Actor loss: maximize advantage
        actor_loss = -(advantage * torch.log(weights_adv.sum(dim=1) + 1e-8)).mean()
        
        # Critic loss: minimize value error
        critic_loss = F.smooth_l1_loss(values, daily_ret)
        
        total_loss = actor_loss + critic_loss
        total_loss.backward()
        
        actor_opt.step()
        critic_opt.step()

print("\nGame loop finished")

# ============================================================================
# [8] FINAL EVALUATION
# ============================================================================
print("\n" + "-" * 80)
print("[6] Final Evaluation")
print("-" * 80)

actor.eval()

# Baseline
print("\nBaseline (Equal Weight):")
w_base = torch.ones(n_assets, device=device) / n_assets
weights_base = []
returns_base = []
with torch.no_grad():
    for t in range(len(latent_states) - REBAL_FREQ):
        weights_base.append(w_base.clone())
        returns_base.append(returns_data[WINDOW + t])

weights_base = torch.stack(weights_base)
returns_base = torch.stack(returns_base)
metrics_base = compute_metrics(weights_base, returns_base)

print(f"  Return:    {metrics_base['return']:7.2f}%")
print(f"  Sharpe:    {metrics_base['sharpe']:7.3f}")
print(f"  Sortino:   {metrics_base['sortino']:7.3f}")
print(f"  Max DD:    {metrics_base['max_dd']:7.2f}%")
print(f"  Win Rate:  {metrics_base['win_rate']:7.1f}%")

# Trained policy
print("\nTrained Policy (Clean):")
weights_policy = []
returns_policy = []
with torch.no_grad():
    for t in range(len(latent_states) - REBAL_FREQ):
        z_t = latent_states[t]
        w_t = actor(z_t.unsqueeze(0)).squeeze()
        weights_policy.append(w_t)
        returns_policy.append(returns_data[WINDOW + t])

weights_policy = torch.stack(weights_policy)
returns_policy = torch.stack(returns_policy)
metrics_policy = compute_metrics(weights_policy, returns_policy)

print(f"  Return:    {metrics_policy['return']:7.2f}%")
print(f"  Sharpe:    {metrics_policy['sharpe']:7.3f}")
print(f"  Sortino:   {metrics_policy['sortino']:7.3f}")
print(f"  Max DD:    {metrics_policy['max_dd']:7.2f}%")
print(f"  Win Rate:  {metrics_policy['win_rate']:7.1f}%")

improvement = metrics_policy['return'] - metrics_base['return']
print(f"\n  Improvement: {improvement:+.2f}pp")

print("\n" + "-" * 80)
print("Stackelberg game complete")
print("-" * 80)

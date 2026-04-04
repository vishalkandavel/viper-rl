import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Dirichlet
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F

torch.manual_seed(42)
np.random.seed(42)

# Data loading

def load_real_data(path="all_stocks.csv", K=10):
    data = pd.read_csv(path)
    data = data[['date', 'Name', 'close']]
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(by='date')

    price_matrix = data.pivot(index='date', columns='Name', values='close')
    price_matrix = price_matrix.dropna(axis=1)

    returns_np = np.log(price_matrix / price_matrix.shift(1)).dropna()
    returns_raw = torch.tensor(returns_np.values, dtype=torch.float32)

    # normalized version for learning
    returns = (returns_raw - returns_raw.mean(dim=0)) / (returns_raw.std(dim=0) + 1e-8)

    T, TOTAL_ASSETS = returns.shape
    idx = torch.randperm(TOTAL_ASSETS)[:K]
    returns = returns[:, idx]
    returns_raw = returns_raw[:, idx]

    print(f"Using {K} assets | Shape: {returns.shape}")
    return returns, returns_raw

returns, returns_raw = load_real_data(K=10)
T, N = returns.shape

# State representation

window = 20

def get_state(returns, t):
    if t < window:
        return torch.zeros(3*N)
    w = returns[t-window:t]
    mean = w.mean(dim=0)
    std = w.std(dim=0)
    last = w[-1]
    return torch.cat([mean, std, last])

states = torch.stack([get_state(returns, t) for t in range(window, T)])

# VAE for latent state representation

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=5):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )
        self.mu = nn.Linear(16, latent_dim)
        self.logvar = nn.Linear(16, latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

vae = VAE(states.shape[1], 5)
opt_vae = optim.Adam(vae.parameters(), lr=1e-3)

print("Initializing VAE...")
for epoch in range(3):
    recon, mu, logvar = vae(states)
    recon_loss = ((recon - states)**2).mean()
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + 0.001 * kl

    opt_vae.zero_grad()
    loss.backward()
    opt_vae.step()

print(f"VAE initialized.")

for p in vae.parameters():
    p.requires_grad = False

# Trading environment

class Env:
    def __init__(self, returns, returns_raw):
        self.returns = returns
        self.returns_raw = returns_raw
        self.T = returns.shape[0]

    def reset(self):
        self.t = window
        return get_state(self.returns, self.t)

    def step(self, action):
        r = self.returns_raw[self.t]
        reward = torch.dot(action, r)

        self.t += 1
        done = self.t >= self.T - 1

        next_state = get_state(self.returns, self.t) if not done else torch.zeros(3*N)
        return next_state, reward, done

# Actor-critic networks

class Actor(nn.Module):
    def __init__(self, input_dim, N):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, N)
        )

    def forward(self, x):
        return F.softplus(self.net(x)) + 0.1

class Critic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# Defender policy with actor-critic

class Defender:
    def __init__(self):
        self.env = Env(returns, returns_raw)
        self.actor = Actor(3*N, N)
        self.critic = Critic(3*N)

        self.opt_a = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.opt_c = optim.Adam(self.critic.parameters(), lr=5e-4)

    def get_perturbed_state(self, raw_state, delta):
        """Perturb state through VAE latent space - used in training (no grads)"""
        with torch.no_grad():
            mu, logvar = vae.encode(raw_state.unsqueeze(0))
        
        z_perturbed = mu + delta.unsqueeze(0)
        
        with torch.no_grad():
            perturbed_state = vae.decode(z_perturbed).squeeze(0)
        
        return perturbed_state
    
    def get_perturbed_state_for_attack(self, raw_state, delta):
        """Perturb state for attack loss (ALLOWS gradient flow through delta)"""
        # Compute mu without gradient
        with torch.no_grad():
            mu, logvar = vae.encode(raw_state.unsqueeze(0))
        
        # Detach mu explicitly and add gradient-enabled delta
        mu = mu.detach()
        z_perturbed = mu + delta.unsqueeze(0)
        
        # Decode with gradient enabled ONLY for delta path
        # (VAE params are frozen so no gradients through decoder anyway)
        perturbed_state = vae.decode(z_perturbed).squeeze(0)
        
        return perturbed_state

    def train(self, delta, episodes=15):
        """Train actor-critic against delta"""
        for ep in range(episodes):
            state = self.env.reset()
            done = False

            while not done:
                s = self.get_perturbed_state(state, delta)
                
                alpha = self.actor(s)
                dist = Dirichlet(alpha)
                action = dist.rsample()
                logp = dist.log_prob(action)

                next_state, reward, done = self.env.step(action)

                # Minimal reward clipping (don't squash signal)
                reward = torch.clamp(reward, -1, 1)

                v = self.critic(s)
                with torch.no_grad():
                    if not done:
                        s_next = self.get_perturbed_state(next_state, delta)
                        v_next = self.critic(s_next)
                    else:
                        v_next = torch.tensor(0.0)

                td = reward + 0.99 * v_next - v

                # CRITIC: Update value function
                self.opt_c.zero_grad()
                critic_loss = td**2
                critic_loss.backward(retain_graph=True)  # KEEP GRAPH for actor
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.opt_c.step()

                # Actor update with entropy regularization for exploration
                self.opt_a.zero_grad()
                entropy = dist.entropy()
                actor_loss = -logp * td.detach() - 0.01 * entropy
                actor_loss.backward(retain_graph=False)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.opt_a.step()

                state = next_state

    def evaluate(self, delta, steps=None):
        """Evaluate policy - returns annualized metrics"""
        if steps is None:
            steps = 500  # Full remaining trajectory
        
        state = self.env.reset()
        rewards = []

        for _ in range(steps):
            s = self.get_perturbed_state(state, delta)
            alpha = self.actor(s)
            action = Dirichlet(alpha).mean  # Deterministic for evaluation
            
            state, reward, done = self.env.step(action)
            rewards.append(reward.detach().item())

            if done:
                break

        rewards = np.array(rewards)
        n_days = len(rewards)
        
        # Annualized return
        log_return = rewards.sum()
        ann_ret = (np.exp(log_return * (252 / n_days)) - 1) * 100
        
        # Sharpe (annualized)
        sharpe = (rewards.mean() / (rewards.std() + 1e-8)) * np.sqrt(252)
        
        # Max drawdown
        cumsum = np.cumsum(rewards)
        peak = np.maximum.accumulate(cumsum)
        dd = cumsum - peak
        max_dd = dd.min()
        
        # Calmar
        calmar = (ann_ret / 100) / (-max_dd + 1e-8) if max_dd < 0 else 0
        
        return ann_ret, sharpe, max_dd, calmar

    def get_attack_loss(self, delta):
        """Compute loss for adversary (gradients flow through delta)"""
        state = self.env.reset()
        rewards = []

        for _ in range(100):  # Shorter window = stronger gradient signal
            s = self.get_perturbed_state_for_attack(state, delta)  # Use gradient-enabled version
            alpha = self.actor(s)
            action = Dirichlet(alpha).rsample()

            state, reward, done = self.env.step(action)
            rewards.append(reward)

            if done:
                break

        if len(rewards) == 0:
            return torch.tensor(0.0, requires_grad=True)
            
        rewards_tensor = torch.stack(rewards)
        total_ret = rewards_tensor.sum()
        
        # Adversary wants to MINIMIZE this
        return total_ret

# Adversary attack mechanism

class Adversary:
    def __init__(self):
        self.lam = 0.00001

    def attack(self, defender, steps=20):
        """Optimize latent perturbation to minimize defender returns"""
        delta = nn.Parameter(torch.randn(5) * 0.1)
        opt = optim.Adam([delta], lr=0.5)

        for s_idx in range(steps):
            opt.zero_grad()
            defender_return = defender.get_attack_loss(delta)
            loss = -defender_return + self.lam * torch.norm(delta)
            loss.backward()
            opt.step()

            with torch.no_grad():
                delta.clamp_(-3.0, 3.0)

        return delta.detach()

# Baseline equal-weight portfolio

def evaluate_equal_weight(env, steps=None):
    """Equal weight (1/N) portfolio baseline"""
    if steps is None:
        steps = 500
    
    state = env.reset()
    rewards = []

    for _ in range(steps):
        action = torch.ones(N) / N
        state, reward, done = env.step(action)
        rewards.append(reward.detach().item())

        if done:
            break

    rewards = np.array(rewards)
    n_days = len(rewards)
    log_return = rewards.sum()
    ann_ret = (np.exp(log_return * (252 / n_days)) - 1) * 100

    return ann_ret

# Stackelberg game loop

def run():
    defender = Defender()
    adv = Adversary()

    print("\nStackelberg Game: Adversarial Portfolio")
    print("-" * 50)

    for iteration in range(1):
        print(f"\nIteration {iteration}")
        
        print("Training clean policy...")
        defender.train(torch.zeros(5), episodes=15)
        
        clean_ret, clean_sharpe, clean_mdd, clean_cal = defender.evaluate(torch.zeros(5))
        bline = evaluate_equal_weight(defender.env)
        
        print(f"Clean:    {clean_ret:7.2f}% | Sharpe: {clean_sharpe:6.3f} | Max DD: {clean_mdd:8.6f}")
        print(f"Baseline: {bline:7.2f}%")
        
        print("Generating attack...")
        delta = adv.attack(defender, steps=20)
        print(f"Delta norm: {delta.norm().item():.4f}")
        
        print("Training robust policy...")
        defender.train(delta, episodes=15)
        
        att_ret, att_sharpe, att_mdd, att_cal = defender.evaluate(delta)
        print(f"Attacked: {att_ret:7.2f}% | Sharpe: {att_sharpe:6.3f} | Max DD: {att_mdd:8.6f}")
        
        perf_drop = clean_ret - att_ret
        print(f"\nPerformance drop: {perf_drop:+.2f}%")
        
        if att_ret > clean_ret:
            print("Result: Attack helped (not expected)")
        else:
            print("Result: Attack reduced performance as expected")

if __name__ == "__main__":
    run()

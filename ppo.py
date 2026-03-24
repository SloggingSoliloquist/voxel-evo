# ppo.py
# Actor-Critic PPO for voxel robot control.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from env import VoxelEnv, OBS_SIZE, ACTION_SIZE

# --- Hyperparameters ---
TOTAL_TIMESTEPS  = 10_000
ROLLOUT_STEPS    = 512       # steps per rollout before update
PPO_EPOCHS       = 4         # gradient update passes per rollout
MINIBATCH_SIZE   = 64
GAMMA            = 0.99      # discount
GAE_LAMBDA       = 0.95      # GAE smoothing
CLIP_EPS         = 0.2       # PPO clip epsilon
ENT_COEF         = 0.01      # entropy bonus
VF_COEF          = 0.5       # value loss coefficient
LR               = 3e-4
MAX_GRAD_NORM    = 0.5


# ------------------------------------------------------------------
# Networks
# ------------------------------------------------------------------

class ActorCritic(nn.Module):
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor_mean  = nn.Linear(64, action_size)
        self.actor_logstd = nn.Parameter(torch.zeros(action_size))
        self.critic      = nn.Linear(64, 1)

    def forward(self, x):
        h = self.trunk(x)
        mean   = torch.sigmoid(self.actor_mean(h))   # [0, 1] → scaled to [0.6,1.6] in env
        std    = torch.exp(self.actor_logstd).expand_as(mean)
        value  = self.critic(h).squeeze(-1)
        return mean, std, value

    def get_action(self, obs_tensor):
        mean, std, value = self(obs_tensor)
        dist    = torch.distributions.Normal(mean, std)
        action  = dist.sample()
        action  = action.clamp(0.0, 1.0)
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

    def evaluate(self, obs_tensor, action_tensor):
        mean, std, value = self(obs_tensor)
        dist     = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action_tensor).sum(-1)
        entropy  = dist.entropy().sum(-1)
        return log_prob, entropy, value


# ------------------------------------------------------------------
# GAE computation
# ------------------------------------------------------------------

def compute_gae(rewards, values, dones, last_value, gamma=GAMMA, lam=GAE_LAMBDA):
    advantages = []
    gae = 0.0
    values_np = values + [last_value]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values_np[t + 1] * (1 - dones[t]) - values_np[t]
        gae   = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


# ------------------------------------------------------------------
# Main training function
# ------------------------------------------------------------------

def train_ppo(morphology, device='cpu'):
    """
    Train a PPO policy for the given morphology.
    Returns (best_mean_reward, final_mean_reward).
    """
    env    = VoxelEnv(morphology)
    policy = ActorCritic(OBS_SIZE, ACTION_SIZE).to(device)
    opt    = optim.Adam(policy.parameters(), lr=LR)

    obs = env.reset()
    total_steps  = 0
    episode_rewards = []
    ep_reward    = 0.0
    best_reward  = -float('inf')

    # Storage for one rollout
    obs_buf     = []
    act_buf     = []
    logp_buf    = []
    rew_buf     = []
    val_buf     = []
    done_buf    = []

    while total_steps < TOTAL_TIMESTEPS:

        # --- Collect rollout ---
        obs_buf.clear()
        act_buf.clear()
        logp_buf.clear()
        rew_buf.clear()
        val_buf.clear()
        done_buf.clear()

        for _ in range(ROLLOUT_STEPS):
            obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action, log_prob, value = policy.get_action(obs_t)

            action_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, done = env.step(action_np)

            obs_buf.append(obs)
            act_buf.append(action_np)
            logp_buf.append(log_prob.item())
            rew_buf.append(reward)
            val_buf.append(value.item())
            done_buf.append(float(done))

            ep_reward   += reward
            total_steps += 1
            obs          = next_obs

            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                obs = env.reset()

        # Record episode-level reward from rollout
        episode_rewards.append(ep_reward)
        ep_reward = 0.0

        # Bootstrap last value
        with torch.no_grad():
            obs_t      = torch.FloatTensor(obs).unsqueeze(0).to(device)
            _, _, last_val = policy(obs_t)
            last_val   = last_val.item()

        advantages, returns = compute_gae(rew_buf, val_buf, done_buf, last_val)

        # Convert to tensors
        obs_t   = torch.FloatTensor(np.array(obs_buf)).to(device)
        act_t   = torch.FloatTensor(np.array(act_buf)).to(device)
        logp_t  = torch.FloatTensor(logp_buf).to(device)
        adv_t   = torch.FloatTensor(advantages).to(device)
        ret_t   = torch.FloatTensor(returns).to(device)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # --- PPO update ---
        n = len(obs_buf)
        for _ in range(PPO_EPOCHS):
            indices = np.random.permutation(n)
            for start in range(0, n, MINIBATCH_SIZE):
                mb_idx  = indices[start:start + MINIBATCH_SIZE]
                mb_obs  = obs_t[mb_idx]
                mb_act  = act_t[mb_idx]
                mb_logp = logp_t[mb_idx]
                mb_adv  = adv_t[mb_idx]
                mb_ret  = ret_t[mb_idx]

                new_logp, entropy, value = policy.evaluate(mb_obs, mb_act)
                ratio    = torch.exp(new_logp - mb_logp)
                surr1    = ratio * mb_adv
                surr2    = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
                actor_loss  = -torch.min(surr1, surr2).mean()
                critic_loss = nn.functional.mse_loss(value, mb_ret)
                ent_loss    = -entropy.mean()
                loss        = actor_loss + VF_COEF * critic_loss + ENT_COEF * ent_loss

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
                opt.step()

        # Track best
        if episode_rewards:
            recent_mean = np.mean(episode_rewards[-10:])
            if recent_mean > best_reward:
                best_reward = recent_mean

    final_mean = float(np.mean(episode_rewards[-10:])) if episode_rewards else 0.0
    return best_reward, final_mean


# ------------------------------------------------------------------
# Policy-returning variant for replay
# ------------------------------------------------------------------

def train_ppo_with_policy(morphology, policy=None, device='cpu'):
    """
    Same as train_ppo but returns (policy, best_reward).
    Used by replay.py to get a trained policy object back.
    """
    env = VoxelEnv(morphology)
    if policy is None:
        policy = ActorCritic(OBS_SIZE, ACTION_SIZE).to(device)
    opt = optim.Adam(policy.parameters(), lr=LR)

    obs = env.reset()
    total_steps     = 0
    episode_rewards = []
    ep_reward       = 0.0
    best_reward     = -float('inf')

    obs_buf  = []
    act_buf  = []
    logp_buf = []
    rew_buf  = []
    val_buf  = []
    done_buf = []

    while total_steps < TOTAL_TIMESTEPS:
        obs_buf.clear(); act_buf.clear(); logp_buf.clear()
        rew_buf.clear(); val_buf.clear(); done_buf.clear()

        for _ in range(ROLLOUT_STEPS):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action, log_prob, value = policy.get_action(obs_t)
            action_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, done = env.step(action_np)

            obs_buf.append(obs); act_buf.append(action_np)
            logp_buf.append(log_prob.item()); rew_buf.append(reward)
            val_buf.append(value.item()); done_buf.append(float(done))

            ep_reward += reward; total_steps += 1; obs = next_obs
            if done:
                episode_rewards.append(ep_reward); ep_reward = 0.0; obs = env.reset()

        episode_rewards.append(ep_reward); ep_reward = 0.0

        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            _, _, last_val = policy(obs_t)
            last_val = last_val.item()

        advantages, returns = compute_gae(rew_buf, val_buf, done_buf, last_val)

        obs_t  = torch.FloatTensor(np.array(obs_buf)).to(device)
        act_t  = torch.FloatTensor(np.array(act_buf)).to(device)
        logp_t = torch.FloatTensor(logp_buf).to(device)
        adv_t  = torch.FloatTensor(advantages).to(device)
        ret_t  = torch.FloatTensor(returns).to(device)
        adv_t  = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n = len(obs_buf)
        for _ in range(PPO_EPOCHS):
            indices = np.random.permutation(n)
            for start in range(0, n, MINIBATCH_SIZE):
                mb_idx  = indices[start:start + MINIBATCH_SIZE]
                mb_obs  = obs_t[mb_idx]; mb_act = act_t[mb_idx]
                mb_logp = logp_t[mb_idx]; mb_adv = adv_t[mb_idx]; mb_ret = ret_t[mb_idx]
                new_logp, entropy, value = policy.evaluate(mb_obs, mb_act)
                ratio  = torch.exp(new_logp - mb_logp)
                surr1  = ratio * mb_adv
                surr2  = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
                loss   = (-torch.min(surr1, surr2).mean()
                          + VF_COEF * nn.functional.mse_loss(value, mb_ret)
                          - ENT_COEF * entropy.mean())
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
                opt.step()

        if episode_rewards:
            recent = np.mean(episode_rewards[-10:])
            if recent > best_reward:
                best_reward = recent

    return policy, best_reward
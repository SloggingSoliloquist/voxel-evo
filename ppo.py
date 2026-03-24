# ppo.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from env import VoxelEnv, OBS_SIZE, ACTION_SIZE

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[PPO] Using device: {DEVICE}")

# --- Hyperparameters ---
TOTAL_TIMESTEPS = 500_000
ROLLOUT_STEPS   = 2048
PPO_EPOCHS      = 10
MINIBATCH_SIZE  = 64
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_EPS        = 0.2
ENT_COEF        = 0.005        # ↓ reduced (less chaotic exploration)
VF_COEF         = 0.5
LR              = 3e-4
MAX_GRAD_NORM   = 0.5


class ActorCritic(nn.Module):
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor_mean   = nn.Linear(64, action_size)

        # ↓ important: lower initial std to prevent jitter
        self.actor_logstd = nn.Parameter(torch.ones(action_size) * -1.5)

        self.critic       = nn.Linear(64, 1)

    def forward(self, x):
        h = self.trunk(x)

        # smoother bounded output (less saturation than sigmoid)
        mean = torch.tanh(self.actor_mean(h)) * 0.5 + 0.5

        std = torch.exp(self.actor_logstd).expand_as(mean)
        value = self.critic(h).squeeze(-1)
        return mean, std, value

    def get_action(self, obs_tensor):
        mean, std, value = self(obs_tensor)

        dist = torch.distributions.Normal(mean, std)

        # sample + squash instead of clamp (important)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action) * 0.5 + 0.5

        log_prob = dist.log_prob(raw_action).sum(-1)
        return action, log_prob, value

    def evaluate(self, obs_tensor, action_tensor):
        mean, std, value = self(obs_tensor)

        # inverse squash (approximate)
        raw_action = torch.atanh(torch.clamp(action_tensor * 2 - 1, -0.999, 0.999))

        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(raw_action).sum(-1)
        entropy  = dist.entropy().sum(-1)
        return log_prob, entropy, value


def compute_gae(rewards, values, dones, last_value):
    advantages = []
    gae = 0.0
    values_np = values + [last_value]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + GAMMA * values_np[t+1] * (1 - dones[t]) - values_np[t]
        gae   = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


def _run_ppo_loop(morphology, policy, opt, device=DEVICE):
    from config import VOXEL_SIZE

    env  = VoxelEnv(morphology)
    obs  = env.reset()
    start_x = env.prev_x

    total_steps     = 0
    episode_rewards = []
    ep_reward       = 0.0
    best_reward     = -float('inf')

    max_x_ever = start_x
    episode_distances = []
    ep_start_x = start_x

    obs_buf  = []; act_buf  = []; logp_buf = []
    rew_buf  = []; val_buf  = []; done_buf = []

    update_num = 0

    while total_steps < TOTAL_TIMESTEPS:

        obs_buf.clear(); act_buf.clear(); logp_buf.clear()
        rew_buf.clear(); val_buf.clear(); done_buf.clear()

        for _ in range(ROLLOUT_STEPS):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)

            with torch.no_grad():
                action, log_prob, value = policy.get_action(obs_t)

            action_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, done = env.step(action_np)

            # --- IMPORTANT: mild smoothness penalty (kills vibration exploit) ---
            if len(act_buf) > 0:
                action_diff = np.mean(np.abs(action_np - act_buf[-1]))
                reward -= 0.05 * action_diff

            obs_buf.append(obs)
            act_buf.append(action_np)
            logp_buf.append(log_prob.item())
            rew_buf.append(reward)
            val_buf.append(value.item())
            done_buf.append(float(done))

            ep_reward   += reward
            total_steps += 1
            obs          = next_obs

            if env.prev_x > max_x_ever:
                max_x_ever = env.prev_x

            if done:
                episode_rewards.append(ep_reward)
                episode_distances.append(env.prev_x - ep_start_x)

                ep_reward  = 0.0
                obs        = env.reset()
                ep_start_x = env.prev_x

        episode_rewards.append(ep_reward)
        episode_distances.append(env.prev_x - ep_start_x)

        ep_reward  = 0.0
        ep_start_x = env.prev_x

        # bootstrap
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

                new_logp, entropy, value = policy.evaluate(obs_t[mb_idx], act_t[mb_idx])

                ratio  = torch.exp(new_logp - logp_t[mb_idx])
                surr1  = ratio * adv_t[mb_idx]
                surr2  = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv_t[mb_idx]

                loss = (
                    -torch.min(surr1, surr2).mean()
                    + VF_COEF * nn.functional.mse_loss(value, ret_t[mb_idx])
                    - ENT_COEF * entropy.mean()
                )

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
                opt.step()

        update_num += 1

        if episode_rewards:
            recent_reward = float(np.mean(episode_rewards[-5:]))
            recent_dist   = float(np.mean(episode_distances[-5:]))

            if recent_reward > best_reward:
                best_reward = recent_reward

            if update_num % 10 == 0:
                print(
                    f"update {update_num} | steps {total_steps}/{TOTAL_TIMESTEPS} "
                    f"| reward {recent_reward:.4f} | dist {recent_dist:.1f}px "
                    f"| best {best_reward:.4f}"
                )

    stats = {
        "best_reward":   best_reward,
        "final_reward":  float(np.mean(episode_rewards[-5:])),
        "final_dist_px": float(np.mean(episode_distances[-5:])),
        "max_dist_px":   float(max_x_ever - start_x),
    }

    return stats


def train_ppo(morphology, device=DEVICE):
    policy = ActorCritic(OBS_SIZE, ACTION_SIZE).to(device)
    opt    = optim.Adam(policy.parameters(), lr=LR)
    stats  = _run_ppo_loop(morphology, policy, opt, device)
    return stats, policy


def continue_ppo(morphology, policy, device=DEVICE):
    policy = policy.to(device)
    opt    = optim.Adam(policy.parameters(), lr=LR * 0.3)
    stats  = _run_ppo_loop(morphology, policy, opt, device)
    return stats, policy
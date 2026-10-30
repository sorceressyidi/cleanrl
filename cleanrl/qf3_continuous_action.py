# QF3 (arXiv 2610.08789) in CleanRL's TD3 harness, as the paper's MuJoCo-v4 experiments (App. A.1, A.3, Table A.2).
# Derived from td3_continuous_action.py: diff the two files to see exactly what changed.
#   1. Actor = flow velocity field v(s, x_tau, phi(tau)): MLP 2x256 SiLU, no LayerNorm, fixed 64-d sinusoidal time
#      embedding, zero-initialised last layer; actions by 5 explicit Euler steps from x_0 = eps ~ N(0, I).
#   2. Actor loss = -Q1(s, x1_clip) + lambda_cfm * L_cfm, x1_clip = a_buf + (1 - tau) clip(v - u_buf, -alpha, alpha)
#      (Eq. 1-4); L_cfm dimension-normalised (Eq. A.2). alpha = inf is the paper's "Direct Q (unclamped)" ablation.
#   3. Actions are stored and learned in the normalised space [-1, 1] (the flow's output space) and rescaled to the
#      env's bounds when executed; exploration and target-smoothing noise as TD3, in that space.
#   4. Evaluation every eval_frequency steps on eval_episodes deterministic episodes (x_0 = 0), written to a JSONL.
#   5. --policy td3 runs CleanRL's own TD3 actor (tanh MLP, -Q1(s, pi(s))) in this exact harness: a control for the
#      harness and the environment stack against CleanRL's published TD3 numbers.
# Everything else (twin critic, target smoothing, delayed actor update, tau, lr, batch, buffer, learning_starts) is
# CleanRL's TD3. Deviations from CleanRL forced by the environment: gymnasium 1.x vector API (SameStep autoreset,
# final_obs), argparse instead of tyro (not installed in this venv).
import argparse
import dataclasses
import json
import math
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    out_dir: str = "runs"
    """directory for tensorboard logs and eval.jsonl"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments (paper: 500k)"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    policy_noise: float = 0.2
    """the scale of policy noise"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25000
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""

    policy: str = "qf3"
    """qf3 (flow actor, QF3 loss) | td3 (CleanRL's TD3 actor in the same harness, control)"""

    # QF3 (paper Table A.2: Hopper 1/0.2, Walker2d 2/0.1, Ant 1/0.5, Humanoid 2/0.5)
    alpha: float = 1.0
    """velocity clip radius; inf = Direct Q (unclamped)"""
    lambda_cfm: float = 0.2
    """conditional flow-matching weight"""
    flow_steps: int = 5
    """Euler steps when sampling actions"""
    time_embed_dim: int = 64
    """sinusoidal time-embedding width"""
    eval_frequency: int = 10000
    """evaluate every this many env steps (paper: 10k)"""
    eval_episodes: int = 5
    """deterministic evaluation episodes (paper: 5)"""


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    for f in dataclasses.fields(Args):
        typ = f.type if isinstance(f.type, type) else eval(f.type)
        if typ is bool:
            p.add_argument(f"--{f.name.replace('_', '-')}", type=lambda s: s.lower() in ("1", "true", "yes"), default=f.default)
        else:
            p.add_argument(f"--{f.name.replace('_', '-')}", type=typ, default=f.default)
    return Args(**vars(p.parse_args()))


def make_env(env_id, seed, idx, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class FlowActor(nn.Module):
    """Velocity field v(s, x_tau, tau) in the normalised action space (App. A.1)."""

    def __init__(self, env, time_embed_dim=64, flow_steps=5):
        super().__init__()
        obs_dim = int(np.array(env.single_observation_space.shape).prod())
        self.act_dim = int(np.prod(env.single_action_space.shape))
        half = time_embed_dim // 2
        k = torch.arange(half, dtype=torch.float32)
        self.register_buffer("freqs", torch.exp(-k * math.log(10000.0) / (half - 1)))
        self.fc1 = nn.Linear(obs_dim + self.act_dim + time_embed_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_v = nn.Linear(256, self.act_dim)
        nn.init.zeros_(self.fc_v.weight)
        nn.init.zeros_(self.fc_v.bias)
        self.flow_steps = flow_steps
        self.register_buffer(
            "action_scale",
            torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32),
        )

    def velocity(self, s, x, t):
        ang = t * self.freqs[None, :]
        h = torch.cat([s, x, torch.sin(ang), torch.cos(ang)], 1)
        h = F.silu(self.fc1(h))
        h = F.silu(self.fc2(h))
        return self.fc_v(h)

    def forward(self, s, deterministic=False):
        """Normalised action x_1 by explicit Euler from x_0 = eps (or 0 when deterministic)."""
        x = torch.zeros(s.shape[0], self.act_dim, device=s.device) if deterministic else torch.randn(s.shape[0], self.act_dim, device=s.device)
        dt = 1.0 / self.flow_steps
        for k in range(self.flow_steps):
            t = torch.full((s.shape[0], 1), k * dt, device=s.device)
            x = x + dt * self.velocity(s, x, t)
        return x

    def to_env(self, a_norm):
        return a_norm * self.action_scale + self.action_bias

    def to_norm(self, a_env):
        return (a_env - self.action_bias) / self.action_scale


class Actor(nn.Module):
    """CleanRL's TD3 actor (td3_continuous_action.py), in the normalised action space [-1, 1]."""

    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.register_buffer("action_scale", torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32))

    def forward(self, x, deterministic=True):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc_mu(x))

    def to_env(self, a_norm):
        return a_norm * self.action_scale + self.action_bias


def evaluate(actor, env_id, seed, episodes, device):
    env = gym.make(env_id)
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, ret = False, 0.0
        while not done:
            with torch.no_grad():
                a = actor(torch.as_tensor(obs, dtype=torch.float32, device=device)[None], deterministic=True)
                a = actor.to_env(a.clamp(-1, 1))[0].cpu().numpy()
            obs, r, term, trunc, _ = env.step(a)
            ret += float(r)
            done = term or trunc
        returns.append(ret)
    env.close()
    return returns


if __name__ == "__main__":

    args = parse_args()
    run_name = (f"{args.env_id}__{args.exp_name}__a{args.alpha}_l{args.lambda_cfm}__{args.seed}" if args.policy == "qf3"
                else f"{args.env_id}__td3_control__{args.seed}")
    out = os.path.join(args.out_dir, run_name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    writer = SummaryWriter(out)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, run_name) for i in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    make_actor = (lambda: FlowActor(envs, args.time_embed_dim, args.flow_steps)) if args.policy == "qf3" else (lambda: Actor(envs))
    actor = make_actor().to(device)
    qf1 = QNetwork(envs).to(device)
    qf2 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    qf2_target = QNetwork(envs).to(device)
    target_actor = make_actor().to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    envs.single_observation_space.dtype = np.float32
    norm_space = gym.spaces.Box(-1.0, 1.0, envs.single_action_space.shape, np.float32)
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        norm_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    eval_log = open(os.path.join(out, "eval.jsonl"), "a")
    start_time = time.time()
    actor_loss = cfm_loss = in_clip = torch.zeros(())

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            a_norm = np.random.uniform(-1.0, 1.0, (envs.num_envs,) + norm_space.shape).astype(np.float32)
        else:
            with torch.no_grad():
                a_norm = actor(torch.Tensor(obs).to(device))
                a_norm += torch.normal(0, args.exploration_noise, size=a_norm.shape, device=device)
                a_norm = a_norm.clamp(-1.0, 1.0).cpu().numpy()
        actions = actor.to_env(torch.as_tensor(a_norm, device=device)).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos and "episode" in infos["final_info"]:
            fin = infos["final_info"]
            for idx in np.flatnonzero(fin.get("_episode", np.ones(envs.num_envs, bool))):
                writer.add_scalar("charts/episodic_return", float(fin["episode"]["r"][idx]), global_step)
                writer.add_scalar("charts/episodic_length", float(fin["episode"]["l"][idx]), global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_obs`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(np.logical_or(terminations, truncations)):
            if d:
                real_next_obs[idx] = infos["final_obs"][idx]
        rb.add(obs, real_next_obs, a_norm, rewards, terminations, [{}] * envs.num_envs)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                clipped_noise = (torch.randn_like(data.actions, device=device) * args.policy_noise).clamp(
                    -args.noise_clip, args.noise_clip
                )
                next_state_actions = (target_actor(data.next_observations) + clipped_noise).clamp(-1.0, 1.0)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0 and args.policy == "td3":
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
            if global_step % args.policy_frequency == 0 and args.policy == "qf3":
                # QF3 actor update (Eq. 1-4): one velocity evaluation at a noised replay action
                a_buf = data.actions
                eps = torch.randn_like(a_buf)
                t = torch.rand(a_buf.shape[0], 1, device=device)
                x_t = (1.0 - t) * eps + t * a_buf
                u_buf = a_buf - eps
                v = actor.velocity(data.observations, x_t, t)
                delta = v - u_buf
                cfm_loss = (delta ** 2).mean()
                x1_clip = a_buf + (1.0 - t) * delta.clamp(-args.alpha, args.alpha)
                actor_loss = -qf1(data.observations, x1_clip).mean() + args.lambda_cfm * cfm_loss
                in_clip = (delta.abs() < args.alpha).float().mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

            if global_step % args.policy_frequency == 0:
                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 1000 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/cfm_loss", cfm_loss.item(), global_step)
                writer.add_scalar("qf3/in_clip_fraction", in_clip.item(), global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if (global_step + 1) % args.eval_frequency == 0:
            rets = evaluate(actor, args.env_id, 10_000 + args.seed * 100, args.eval_episodes, device)
            row = dict(step=global_step + 1, eval_return=float(np.mean(rets)), returns=rets,
                       qf_loss=float(qf_loss.item() / 2.0) if global_step > args.learning_starts else None,
                       cfm_loss=float(cfm_loss.item()), in_clip=float(in_clip.item()),
                       sps=int(global_step / (time.time() - start_time)), wall_s=round(time.time() - start_time, 1))
            eval_log.write(json.dumps(row) + "\n"); eval_log.flush()
            writer.add_scalar("eval/return", row["eval_return"], global_step + 1)
            print(json.dumps(row), flush=True)

    envs.close()
    writer.close()
    eval_log.close()

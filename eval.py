import os
from copy import deepcopy

from torch import nn


import d4rl
import gym

import numpy as np
import torch

from cleandiffuser.diffusion import DiscreteDiffusionSDE, ContinuousDiffusionSDE
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import IDQLMlp, SfBCUNet, MPPOUNet, DQLMlp
from cleandiffuser.utils import report_parameters, DQLCritic

from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset

from cleandiffuser.nn_condition import MLPCondition
from cleandiffuser.nn_diffusion import DiT1d
from d4rl.ope import normalize
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cbook as cbook


def load_model(mode, obs_dim, act_dim, args):
    if mode == "critic":
        q = DDAQCritic(obs_dim, act_dim, hidden_dim=args.critic_hidden_dim).to(args.device)
        q_target = deepcopy(iql_q).requires_grad_(False).eval()
        save_path = f'results/mppo/critic/{args.task.env_name}/MPPOUNet/'
        # 添加路径验证
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"路径不存在: {save_path}")
        critic_ckpt = torch.load(save_path + f"critic_ckpt_latest.pt")
        q.load_state_dict(critic_ckpt["critic"])
        q_target.load_state_dict(critic_ckpt["critic_target"])
        return q, q_target
    elif mode == 'policy':
        nn_diffusion = DDAQMlp(obs_dim, act_dim, emb_dim=64, timestep_emb_type="positional").to(args.device)
        nn_condition = IdentityCondition(dropout=0.0).to(args.device)

        agent = DiscreteDiffusionSDE(
            nn_diffusion, nn_condition, ema_rate=args.ema_rate,
            device=args.device, optim_params={"lr": args.actor_learning_rate},
            x_max=+1. * torch.ones((1, act_dim)),
            x_min=-1. * torch.ones((1, act_dim)), diffusion_steps=args.diffusion_steps
        )
        save_path = f'results/mppo/policy/{args.task.env_name}/MPPOUNet/'
        # 添加路径验证
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"路径不存在: {save_path}")
        policy.load(save_path + f"diffusion_ckpt_latest.pt")
        return policy


def eval_policy(env_name, train_step, sampling_steps, num_candidates, dataset):
    device = 'cuda:0'
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim
    # load model
    save_path = f'results/dmpo/{env_name}/DIT/'
    critic_ckpt = torch.load(save_path + f"critic_ckpt_{train_step}.pt")
    iql_q = DQLCritic(obs_dim, act_dim, hidden_dim=256).to(device)
    q = deepcopy(iql_q).requires_grad_(False).eval()
    q.load_state_dict(critic_ckpt["critic_target"])
    q_target = q
    nn_diffusion = DQLMlp(obs_dim, act_dim, emb_dim=64, timestep_emb_type="positional").to(device)
    nn_condition = IdentityCondition(dropout=0.0).to(device)
    agent = DiscreteDiffusionSDE(
        nn_diffusion, nn_condition, ema_rate=0.995,
        device=device, optim_params={"lr": 0.0003},
        x_max=+1. * torch.ones((1, act_dim)),
        x_min=-1. * torch.ones((1, act_dim)), diffusion_steps=5
    )
    agent.load(save_path + f"diffusion_ckpt_{train_step}.pt")

    agent.eval()
    q_target.eval()
    env_eval = gym.vector.make(env_name, 50)
    normalizer = dataset.get_normalizer()
    episode_rewards = []
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim
    prior = torch.zeros((50 * num_candidates, act_dim), device=device)
    for i in range(3):

        obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

        while not np.all(cum_done) and t < 1000 + 1:
            # normalize obs
            obs = torch.tensor(normalizer.normalize(obs), device=device, dtype=torch.float32)
            obs = obs.unsqueeze(1).repeat(1, num_candidates, 1).view(-1, obs_dim)

            # sample trajectories
            action, log = agent.sample(
                prior,
                solver='ddpm',
                n_samples=50 * num_candidates,
                sample_steps=sampling_steps,
                condition_cfg=obs,
                use_ema=True, w_cfg=1., temperature=0.5)
            act = action

            # resample
            with torch.no_grad():
                q = torch.min(*q_target(obs, act))
                q = q.view(-1, num_candidates, 1)
                w = torch.softmax(q, 1)
                act = act.view(-1, num_candidates, act_dim)

                indices = torch.multinomial(w.squeeze(-1), 1).squeeze(-1)
                sampled_act = act[torch.arange(act.shape[0]), indices].cpu().numpy()

            # step
            obs, rew, done, info = env_eval.step(sampled_act)

            t += 1
            cum_done = done if cum_done is None else np.logical_or(cum_done, done)
            ep_reward += (rew * (1 - cum_done)) if t < 1000 else rew

        episode_rewards.append(ep_reward)
    episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
    episode_rewards = np.array(episode_rewards)
    return np.mean(episode_rewards), np.std(episode_rewards)


def graph(env, line_type, sampling_steps, num_candidates):
    line_types = np.repeat(line_type, 11).tolist()
    train_steps = [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000]
    x = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    mode = 'num'
    for e in env:
        scores = []
        env = gym.make(e)
        dataset = D4RLMuJoCoDataset(
            env.get_dataset(), horizon=1, terminal_penalty=-100,
            discount=0.99)
        if mode == 'step':
            for s in sampling_steps:
                for t in train_steps:
                    if t == 0:
                        scores.append(0)
                    else:
                        score, _ = eval_policy(e, t, s, 50, dataset)
                        scores.append(score)

        elif mode == 'num':
            for n in num_candidates:
                for t in train_steps:
                    if t == 0:
                        scores.append(0)
                    else:
                        score, _ = eval_policy(e, t, 5, n, dataset)
                        scores.append(score)

        data = {
            'x': x * len(sampling_steps),
            'y': scores,
            'line_type': line_types

        }
        print(len(data['x']), len(data['y']), len(data['line_type']))
        df = pd.DataFrame(data)
        sns.lineplot(x='x', y='y', hue='line_type', data=df, markers=True)  # 添加 markers 使线条更容易区分
        plt.title(f"{e}")
        plt.xlabel("Train Progress (# 20k steps)")
        plt.ylabel("Normalized Score")
        plt.grid(True)
        plt.show()


if not hasattr(cbook, '_Stack') and hasattr(cbook, 'Stack'):
    cbook._Stack = cbook.Stack

if __name__ == "__main__":
    env = ['halfcheetah-medium-expert-v2', 'hopper-medium-expert-v2', 'walker2d-medium-expert-v2']
    sampling_steps = [3, 5, 15, 25]
    num_candidates = [1, 20, 50, 100]
    #line_type = ['T=3', 'T=5', 'T=15', 'T=25']
    line_type = ['N=1', 'N=20', 'N=50', 'N=100']
    graph(env, line_type, sampling_steps, num_candidates)

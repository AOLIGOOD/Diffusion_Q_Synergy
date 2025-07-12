import hydra
import logging
import math
import os
import random
import argparse
from copy import deepcopy
from typing import Union, Callable, Optional

import cv2
import gym
import numpy as np
import torch
from gym.wrappers import RecordVideo
from torch import nn
from torch.nn import LayerNorm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import wandb

import d4rl
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import ContinuousDiffusionSDE, DiscreteDiffusionSDE
from cleandiffuser.nn_condition import IdentityCondition, BaseNNCondition
from cleandiffuser.nn_diffusion import BaseNNDiffusion, DQLMlp
from cleandiffuser.utils import DQLCritic, FreezeModules, at_least_ndim
from pipelines.utils import set_seed
import torch.nn.functional as F



@hydra.main(config_path='configs/critic', config_name="mujoco", version_base=None)
def train_without_log_eval(args):
    import time
    start_time = time.time()
    set_seed(args.seed)
    save_path = f'results/dmpo/{args.task.env_name}/event/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    
    # Create Dataset
    env = gym.make(args.env_name)
    dataset = D4RLMuJoCoTDDataset(d4rl.qlearning_dataset(env), normalize_reward=True)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    # Initialize Critic
    critic = DQLCritic(obs_dim, act_dim, hidden_dim=args.critic_hidden_dim).to(args.device)
    critic_target = deepcopy(critic).requires_grad_(False).eval()
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_learning_rate)

    # Initialize Diffusion Model
    nn_diffusion = EventMlp(
        obs_dim, 
        act_dim, 
        emb_dim=64,
        timestep_emb_type="positional"
    ).to(args.device)
    nn_condition = IdentityCondition(dropout=0.0).to(args.device)

    agent = DiscreteDiffusionSDE(
        nn_diffusion, nn_condition, ema_rate=args.ema_rate,
        device=args.device, optim_params={"lr": args.actor_learning_rate},
        x_max=+1. * torch.ones((1, act_dim)),
        x_min=-1. * torch.ones((1, act_dim)), diffusion_steps=args.diffusion_steps,
    )

    # Training Loop
    q_lr_scheduler = CosineAnnealingLR(critic_optim, T_max=args.gradient_steps)
    actor_lr_scheduler = CosineAnnealingLR(agent.optimizer, T_max=args.gradient_steps)

    agent.train()
    critic.train()

    n_gradient_step = 0
    prior = torch.zeros((args.batch_size, act_dim), device=args.device)
    for batch in loop_dataloader(dataloader):
        obs, next_obs = batch["obs"]["state"].to(args.device), batch["next_obs"]["state"].to(args.device)
        act = batch["act"].to(args.device)
        rew = batch["rew"].to(args.device)
        tml = batch["tml"].to(args.device)

        # Critic Training
        current_q1, current_q2 = critic(obs, act)
        next_act, _ = agent.sample(
            prior, solver=args.solver,warm_start_reference=act,warm_start_forward_level=0.4,
            n_samples=args.batch_size, sample_steps=2, use_ema=True,
            temperature=1.0, condition_cfg=next_obs, w_cfg=1.0, requires_grad=False)
        target_q = torch.min(*critic_target(next_obs, next_act))
        target_q = (rew + (1 - tml) * args.discount * target_q).detach()
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        critic_optim.step()
        q_lr_scheduler.step()

        # Policy Training
        pred_act, _, = agent.sample(
            act, solver=args.solver,warm_start_reference=act,warm_start_forward_level=0.4,
            n_samples=args.batch_size, sample_steps=2, use_ema=False,
            temperature=1.0, condition_cfg=obs, w_cfg=1.0, requires_grad=True)
        
        with FreezeModules([critic, ]):
            q_new_action = torch.min(*critic(obs, pred_act))
            q_data_action = torch.min(*critic(obs, act))
        adv_loss = -(q_new_action.mean() / q_data_action.abs().mean().detach())
        bc_loss = agent.loss(act, obs)
        loss = 1.0 * adv_loss + bc_loss

        agent.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.model.parameters(), agent.grad_clip_norm) \
            if agent.grad_clip_norm else None
        agent.optimizer.step()
        actor_lr_scheduler.step()

        # EMA Update
        if n_gradient_step % args.ema_update_interval == 0:
            if n_gradient_step >= 1000:
                agent.ema_update()
            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(args.ema_rate * param.data + (1 - args.ema_rate) * target_param.data)

        # Model Saving
        if (n_gradient_step + 1) % args.save_interval == 0:
            torch.save({
                "critic": critic.state_dict(),
                "critic_target": critic_target.state_dict(),
            }, save_path + f"critic_ckpt_{n_gradient_step + 1}.pt")
            torch.save({
                "critic": critic.state_dict(),
                "critic_target": critic_target.state_dict(),
            }, save_path + f"critic_ckpt_latest.pt")
            agent.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
            agent.save(save_path + f"diffusion_ckpt_latest.pt")

        n_gradient_step += 1
        if n_gradient_step >= args.gradient_steps:
            break
    
    end_time = time.time()
    total_time = end_time - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Training completed in: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

def train_sweep():
    # Define the training function for sweep
    def train():
        # Initialize wandb run for this sweep iteration
        wandb.init()
        
        # Create args object from wandb config
        args = parse_args()
        
        # Override args with wandb config values
        args.discount = wandb.config.get('discount', args.discount)
        args.normalize_reward = wandb.config.get('normalize_reward', args.normalize_reward)
        args.solver = wandb.config.get('solver', args.solver)
        args.sampling_steps = wandb.config.get('sampling_steps', args.sampling_steps)
        args.actor_learning_rate = wandb.config.get('actor_learning_rate', args.actor_learning_rate)
        args.batch_size = wandb.config.get('batch_size', args.batch_size)
        args.critic_hidden_dim = wandb.config.get('critic_hidden_dim', args.critic_hidden_dim)
        args.critic_learning_rate = wandb.config.get('critic_learning_rate', args.critic_learning_rate)
        args.temperature = wandb.config.get('temperature', args.temperature)
        args.ema_rate = wandb.config.get('ema_rate', args.ema_rate)
        args.num_candidates = wandb.config.get('num_candidates', args.num_candidates)
        args.discretization = wandb.config.get('discretization', args.discretization)
        args.noise_schedule = wandb.config.get('noise_schedule', args.noise_schedule)
        args.ema_update_interval = wandb.config.get('ema_update_interval', args.ema_update_interval)
        args.predict_noise = wandb.config.get('predict_noise', args.predict_noise)
        args.log_interval = wandb.config.get('log_interval', args.log_interval)
        args.emb_dim = wandb.config.get('emb_dim', args.emb_dim)
        args.agent_dim = wandb.config.get('agent_dim', args.agent_dim)
        args.adv_loss_coef = wandb.config.get('adv_loss_coef', args.adv_loss_coef)
        
        # Run training with the configured args
        train2(args)
    
    # Create and run the sweep
    import yaml
    with open('sweep.yaml') as f:
        sweep_config = yaml.safe_load(f)
    sweep_id = wandb.sweep(sweep=sweep_config, project='DQS_gym')
    wandb.agent(sweep_id, function=train)

def parse_args():
    parser = argparse.ArgumentParser()
    # Basic parameters
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--env_name', type=str, default='walker2d-medium-expert-v2')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--normalize_reward', type=bool, default=True)
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--gradient_steps', type=int, default=200000)
    parser.add_argument('--log_interval', type=int, default=1000)
    parser.add_argument('--save_interval', type=int, default=100000)
    parser.add_argument('--num_envs', type=int, default=50)
    parser.add_argument('--num_episodes', type=int, default=3)
    parser.add_argument('--num_candidates', type=int, default=5)
    
    # Critic parameters
    parser.add_argument('--critic_hidden_dim', type=int, default=256)
    parser.add_argument('--critic_learning_rate', type=float, default=0.0003)
    
    # Agent parameters
    parser.add_argument('--actor_learning_rate', type=float, default=0.0003)
    parser.add_argument('--ema_rate', type=float, default=0.995)
    parser.add_argument('--ema_update_interval', type=int, default=5)
    parser.add_argument('--gn', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--use_ema', type=bool, default=True)
    parser.add_argument('--predict_noise', type=bool, default=True)
    parser.add_argument('--adv_loss_coef', type=float, default=1.0)
    #parser.add_argument('--event_loss_coef', type=float, default=1.0)
    
    # Diffusion parameters
    parser.add_argument('--solver', type=str, default='ddpm')
    parser.add_argument('--diffusion_steps', type=int, default=5)
    parser.add_argument('--sampling_steps', type=int, default=5)
    parser.add_argument('--discretization', type=str, default='uniform')
    parser.add_argument('--noise_schedule', type=str, default='cosine')
    parser.add_argument('--emb_dim', type=int, default=16)
    parser.add_argument('--agent_dim', type=int, default=256)
    parser.add_argument('--reward_tune', type=str, default='no')
    
    args = parser.parse_args()
    
    # Set environment-specific defaults
    env_defaults = {
        'halfcheetah-medium-v2':         {'gn': 9.0,  'top_k': 1},
        'hopper-medium-v2':              {'gn': 9.0,  'top_k': 2},
        'walker2d-medium-v2':            {'gn': 1.0,  'top_k': 1},
        'halfcheetah-medium-replay-v2':  {'gn': 2.0,  'top_k': 0},
        'hopper-medium-replay-v2':       {'gn': 4.0,  'top_k': 2},
        'walker2d-medium-replay-v2':     {'gn': 4.0,  'top_k': 1},
        'halfcheetah-medium-expert-v2':  {'gn': 7.0,  'top_k': 0},
        'hopper-medium-expert-v2':       {'gn': 5.0,  'top_k': 2},
        'walker2d-medium-expert-v2':     {'gn': 5.0,  'top_k': 1},
        'kitchen-complete-v0':           {'gn': 9.0,  'top_k': 2, 'adv_loss_coef': 0.005, 'num_epochs': 250},
        'kitchen-partial-v0':            {'gn': 10.0, 'top_k': 2, 'adv_loss_coef': 0.005, 'num_epochs': 1000},
        'kitchen-mixed-v0':              {'gn': 10.0, 'top_k': 0, 'adv_loss_coef': 0.005, 'num_epochs': 1000}
    }
    
    if args.env_name in env_defaults:
        for key, value in env_defaults[args.env_name].items():
            if getattr(args, key) == parser.get_default(key):  # Only override if using default value
                setattr(args, key, value)
    
    return args

def train2():
    args = parse_args()
    set_seed(args.seed)
    wandb.init(project='DQS_gym', name=args.env_name, config=vars(args))
    save_path = f'results/DQS/{args.env_name}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    # ---------------------- Create Dataset ----------------------
    env = gym.make(args.env_name)
    dataset = D4RLMuJoCoTDDataset(d4rl.qlearning_dataset(env), normalize_reward=args.normalize_reward)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    # ------------------ Critic ---------------------
    critic = DQLCritic(obs_dim, act_dim, hidden_dim=args.critic_hidden_dim).to(args.device)
    critic_target = deepcopy(critic).requires_grad_(False).eval()
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_learning_rate)

    # --------------- Diffusion Model --------------------
    
    nn_diffusion = DQLMlp(obs_dim, act_dim, emb_dim=16, timestep_emb_type="positional").to(args.device)
    nn_condition = IdentityCondition(dropout=0.0).to(args.device)

    agent = DiscreteDiffusionSDE(
        nn_diffusion, nn_condition, ema_rate=args.ema_rate, noise_schedule=args.noise_schedule,
        device=args.device, optim_params={"lr": args.actor_learning_rate},
        x_max=+1. * torch.ones((1, act_dim)), discretization=args.discretization,
        x_min=-1. * torch.ones((1, act_dim)), diffusion_steps=args.diffusion_steps,
        grad_clip_norm=args.gn  # Use gn value for gradient clipping
    )

    # ---------------------- Training ----------------------
    q_lr_scheduler = CosineAnnealingLR(critic_optim, T_max=args.gradient_steps)
    actor_lr_scheduler = CosineAnnealingLR(agent.optimizer, T_max=args.gradient_steps)

    agent.train()
    critic.train()

    n_gradient_step = 0
    log = {"policy_loss": 0., "q_loss": 0., "bc_loss": 0.,'target_q_mean': 0.}
    prior = torch.zeros((args.batch_size, act_dim), device=args.device)
    for batch in loop_dataloader(dataloader):
        obs, next_obs = batch["obs"]["state"].to(args.device), batch["next_obs"]["state"].to(args.device)
        act = batch["act"].to(args.device)
        rew = batch["rew"].to(args.device)
        tml = batch["tml"].to(args.device)

        # Critic Training
        current_q1, current_q2 = critic(obs, act)

        next_act, _ = agent.sample(
            prior, solver=args.solver,
            n_samples=args.batch_size, sample_steps=5, use_ema=True,
            temperature=1.0, condition_cfg=next_obs, w_cfg=1.0, requires_grad=False)

        target_q = torch.min(*critic_target(next_obs, next_act))
        target_q = (rew + (1 - tml) * args.discount * target_q).detach()

        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), args.gn)
        critic_optim.step()

        q_lr_scheduler.step()
        

        # Policy Training
        pred_act, _ = agent.sample(
            act,solver=args.solver,warm_start_reference=act,warm_start_forward_level=0.4,
            n_samples=args.batch_size, sample_steps=2, use_ema=False,
            temperature=1.0, condition_cfg=obs, w_cfg=1.0, requires_grad=True)

        with FreezeModules([critic, ]):
            q1_new_action, q2_new_action = critic(obs, pred_act)
        if np.random.uniform() > 0.5:
            adv_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
        else:
            adv_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
        
        
        bc_loss = agent.loss(act, obs)
        loss = args.adv_loss_coef * adv_loss + bc_loss
        
        agent.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.model.parameters(), args.gn)  # Use gn value for gradient clipping
        agent.optimizer.step()

        actor_lr_scheduler.step()

        # -- ema
        if n_gradient_step % args.ema_update_interval == 0:
            if n_gradient_step >= 1000:
                agent.ema_update()
            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(args.ema_rate * param.data + (1 - args.ema_rate) * target_param.data)
        #  ----------- Logging ------------
        log["q_loss"] += critic_loss.item()
        log["policy_loss"] += loss.item()
        log["target_q_mean"] += target_q.mean().item()
        log["bc_loss"] += bc_loss.item()

        if (n_gradient_step + 1) % args.log_interval == 0:
            log["gradient_steps"] = n_gradient_step + 1
            log["q_loss"] /= args.log_interval
            log["target_q_mean"] /= args.log_interval
            log["policy_loss"] /= args.log_interval
            log["bc_loss"] /= args.log_interval
            log['avg_reward_mean'], log['avg_reward_std'] = eval_policy(agent, critic_target, dataset, args, env)
            wandb.log(log)
            log = {"policy_loss": 0., "q_loss": 0., "bc_loss": 0., 'target_q_mean': 0.}

        # ----------- Saving ------------
        if (n_gradient_step + 1) % args.save_interval == 0:
            torch.save({
                "critic": critic.state_dict(),
                "critic_target": critic_target.state_dict(),
            }, save_path + f"critic_ckpt_{n_gradient_step + 1}.pt")
            torch.save({
                "critic": critic.state_dict(),
                "critic_target": critic_target.state_dict(),
            }, save_path + f"critic_ckpt_latest.pt")
            agent.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
            agent.save(save_path + f"diffusion_ckpt_latest.pt")

        n_gradient_step += 1
        if n_gradient_step >= args.gradient_steps:
            break


def eval_policy(agent, q_target, dataset, args, env):
    agent.eval()
    q_target.eval()
    env_eval = gym.vector.make(args.env_name, args.num_envs)
    normalizer = dataset.get_normalizer()
    episode_rewards = []
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim
    prior = torch.zeros((args.num_envs * args.num_candidates, act_dim), device=args.device)
    for i in range(args.num_episodes):

        obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

        while not np.all(cum_done) and t < 1000 + 1:
            # normalize obs
            obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
            obs = obs.unsqueeze(1).repeat(1, args.num_candidates, 1).view(-1, obs_dim)

            # sample trajectories
            act, _, = agent.sample(
                prior,
                solver=args.solver,
                n_samples=args.num_envs * args.num_candidates,
                sample_steps=5,
                condition_cfg=obs, w_cfg=1.0,
                use_ema=True, temperature=args.temperature)

            # resample
            with torch.no_grad():
                q = torch.min(*q_target(obs, act))
                q = q.view(-1, args.num_candidates, 1)
                w = torch.softmax(q, 1)
                act = act.view(-1, args.num_candidates, act_dim)

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



if __name__ == "__main__":
    train2()
    #train_sweep()
    #train_without_log_eval()

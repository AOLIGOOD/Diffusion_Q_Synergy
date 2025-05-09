import logging
import math
import os
from copy import deepcopy

import gym
import hydra
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import d4rl
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import DDAQMlp
from cleandiffuser.utils import FreezeModules, DDAQCritic

import torch.nn.functional as F


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(config_path='/configs/critic', config_name="mujoco", version_base=None)
def train(args, ):
    set_seed(args.seed)
    save_path = f'results/dmpo/{args.task.env_name}/DIT/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    # ---------------------- Create Dataset ----------------------
    env = gym.make(args.task.env_name)
    dataset = D4RLMuJoCoTDDataset(d4rl.qlearning_dataset(env), normalize_reward=True)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    # ------------------ Critic ---------------------
    critic = DDAQCritic(obs_dim, act_dim, hidden_dim=args.critic_hidden_dim).to(args.device)
    critic_target = deepcopy(critic).requires_grad_(False).eval()
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_learning_rate)

    # --------------- Diffusion Model --------------------
    nn_diffusion = DDAQMlp(obs_dim, act_dim, emb_dim=64, timestep_emb_type="positional").to(args.device)
    nn_condition = IdentityCondition(dropout=0.0).to(args.device)

    agent = DiscreteDiffusionSDE(
        nn_diffusion, nn_condition, ema_rate=args.ema_rate,
        device=args.device, optim_params={"lr": args.actor_learning_rate},
        x_max=+1. * torch.ones((1, act_dim)),
        x_min=-1. * torch.ones((1, act_dim)), diffusion_steps=args.diffusion_steps
    )

    # ---------------------- Training ----------------------
    q_lr_scheduler = CosineAnnealingLR(critic_optim, T_max=args.gradient_steps)
    actor_lr_scheduler = CosineAnnealingLR(agent.optimizer, T_max=args.gradient_steps)

    agent.train()
    critic.train()

    n_gradient_step = 0
    log = {"policy_loss": 0., "q_loss": 0., "bc_loss": 0., 'target_q_mean': 0.}
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
            n_samples=args.batch_size, sample_steps=args.sampling_steps, use_ema=True,
            temperature=1.0, condition_cfg=next_obs, w_cfg=1.0, requires_grad=False)

        target_q = torch.min(*critic_target(next_obs, next_act))
        target_q = (rew + (1 - tml) * args.discount * target_q).detach()

        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        q_lr_scheduler.step()

        # Policy Training

        new_act, _, = agent.sample(
            prior, solver=args.solver,
            n_samples=args.batch_size, sample_steps=args.sampling_steps, use_ema=False,
            temperature=1.0, condition_cfg=obs, w_cfg=1.0, requires_grad=True)

        with FreezeModules([critic, ]):
            q_new_action = torch.min(*critic(obs, new_act))
            q_data_action = torch.min(*critic(obs, act))
        adv_loss = (-q_new_action.mean() / q_data_action.abs().mean().detach())

        adv_loss = torch.clamp(adv_loss, min=-1.1, max=1.1)

        bc_loss = agent.loss(act, obs)
        loss = adv_loss + bc_loss

        agent.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.model.parameters(), agent.grad_clip_norm) \
            if agent.grad_clip_norm else None
        agent.optimizer.step()

        actor_lr_scheduler.step()

        # -- ema
        if n_gradient_step % args.ema_update_interval == 0:
            if n_gradient_step >= 1000:
                agent.ema_update()
            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(0.995 * param.data + (1 - 0.995) * target_param.data)
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


if __name__ == "__main__":
    eval_policy_trained_std()

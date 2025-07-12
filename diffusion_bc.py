import os
from copy import deepcopy

import d4rl
import gym
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import DQLMlp
from cleandiffuser.utils import report_parameters, DQLCritic, FreezeModules
from utils import set_seed


@hydra.main(config_path="../configs/critic", config_name="mujoco", version_base=None)
def pipeline(args):

    set_seed(args.seed)

    save_path = f'/home/ao/Desktop/CleanDiffuser/results/{args.pipeline_name}/{args.task.env_name}/'
    os.makedirs(save_path, exist_ok=True)

    # ---------------------- Create Dataset ----------------------
    env = gym.make(args.task.env_name)
    dataset = D4RLMuJoCoTDDataset(d4rl.qlearning_dataset(env), args.normalize_reward)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    # --------------- Network Architecture -----------------
    nn_diffusion = DQLMlp(obs_dim, act_dim, emb_dim=64, timestep_emb_type="positional").to(args.device)
    nn_condition = IdentityCondition(dropout=0.0).to(args.device)

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"==============================================================================")

    # --------------- Diffusion Model Actor --------------------
    actor = DiscreteDiffusionSDE(
        nn_diffusion, nn_condition, predict_noise=args.predict_noise, optim_params={"lr": args.actor_learning_rate},
        x_max=+1. * torch.ones((1, act_dim), device=args.device),
        x_min=-1. * torch.ones((1, act_dim), device=args.device),
        diffusion_steps=args.diffusion_steps, ema_rate=args.ema_rate, device=args.device)

    # ------------------ Critic ---------------------
    critic = DQLCritic(obs_dim, act_dim, hidden_dim=args.hidden_dim).to(args.device)
    critic_target = deepcopy(critic).requires_grad_(False).eval()
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_learning_rate)

    # ---------------------- Training ----------------------
    if args.mode == "train":

        actor_lr_scheduler = CosineAnnealingLR(actor.optimizer, T_max=args.gradient_steps)
        critic_lr_scheduler = CosineAnnealingLR(critic_optim, T_max=args.gradient_steps)

        actor.train()
        critic.train()

        n_gradient_step = 0
        log = {"bc_loss": 0., "critic_loss": 0., "target_q_mean": 0.}

        prior = torch.zeros((args.batch_size, act_dim), device=args.device)

        for batch in loop_dataloader(dataloader):

            obs, next_obs = batch["obs"]["state"].to(args.device), batch["next_obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)
            rew = batch["rew"].to(args.device)
            tml = batch["tml"].to(args.device)

            # Critic Training
            current_q1, current_q2 = critic(obs, act)

            next_act, _ = actor.sample(
                prior, solver=args.solver,
                n_samples=args.batch_size, sample_steps=args.sampling_steps, use_ema=True,
                temperature=1.0, condition_cfg=next_obs, w_cfg=1.0, requires_grad=False)

            target_q = torch.min(*critic_target(next_obs, next_act))
            target_q = (rew + (1 - tml) * args.discount * target_q).detach()

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

            # -- Policy Training (Only BC loss)
            bc_loss = actor.loss(act, obs)

            actor.optimizer.zero_grad()
            bc_loss.backward()
            actor.optimizer.step()

            actor_lr_scheduler.step()
            critic_lr_scheduler.step()

            # -- ema
            if n_gradient_step % args.ema_update_interval == 0:
                if n_gradient_step >= 1000:
                    actor.ema_update()
                for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                    target_param.data.copy_(0.995 * param.data + (1 - 0.995) * target_param.data)

            # Logging
            log["bc_loss"] += bc_loss.item()
            log["critic_loss"] += critic_loss.item()
            log["target_q_mean"] += target_q.mean().item()

            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["bc_loss"] /= args.log_interval
                log["critic_loss"] /= args.log_interval
                log["target_q_mean"] /= args.log_interval
                print(log)
                log = {"bc_loss": 0., "critic_loss": 0., "target_q_mean": 0.}

            # Saving
            if (n_gradient_step + 1) % args.save_interval == 0:
                actor.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
                actor.save(save_path + f"diffusion_ckpt_latest.pt")
                torch.save({
                    "critic": critic.state_dict(),
                    "critic_target": critic_target.state_dict(),
                }, save_path + f"critic_ckpt_{n_gradient_step + 1}.pt")
                torch.save({
                    "critic": critic.state_dict(),
                    "critic_target": critic_target.state_dict(),
                }, save_path + f"critic_ckpt_latest.pt")

            n_gradient_step += 1
            if n_gradient_step >= args.gradient_steps:
                break

    # ---------------------- Evaluate Diffusion Process ----------------------
    elif args.mode == "inference":

        actor.load(save_path + f"diffusion_ckpt_{args.ckpt}.pt")
        critic_ckpt = torch.load(save_path + f"critic_ckpt_{args.ckpt}.pt")
        critic.load_state_dict(critic_ckpt["critic"])
        critic_target.load_state_dict(critic_ckpt["critic_target"])

        actor.eval()
        critic.eval()
        critic_target.eval()

        env_eval = gym.vector.make(args.task.env_name, args.num_envs)
        normalizer = dataset.get_normalizer()
        step_rewards = []  # 记录每个扩散步的归一化奖励
        all_diffs = []     # 记录所有episode的差值
        prior = torch.zeros((args.num_envs, act_dim), device=args.device)
        for i in range(6):

            obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

            while not np.all(cum_done) and t < 1000 + 1:
                # normalize obs
                obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
            

                # sample actions
                _, log = actor.sample(
                    prior, preserve_history=True,
                    solver=args.solver,
                    n_samples=args.num_envs,
                    sample_steps=5,
                    condition_cfg=obs, w_cfg=1.0,  # 确保condition是2D
                    use_ema=args.use_ema, temperature=args.temperature)

                # get actions for current diffusion step
                sampled_act = log["sample_history"][0, i]
                
                # step - ensure action has correct shape (num_envs, action_dim)
                sampled_act = sampled_act.reshape(args.num_envs, -1)
                obs, rew, done, info = env_eval.step(sampled_act)

                t += 1
                cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                ep_reward += (rew * (1 - cum_done)) if t < 1000 else rew
                

                if np.all(cum_done):
                    break

            step_rewards.append(ep_reward)

            step_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in step_rewards]
            
            # 计算每个step的平均奖励
            mean_rewards = np.mean(step_rewards, axis=1)
            
            # 计算相邻步数差值
            if len(mean_rewards) > 1:
                diffs = np.diff(mean_rewards)
                all_diffs.append(diffs)

        
        print(mean_rewards)
        # 最终可视化（所有步数完成后执行一次）
        if len(mean_rewards) == 6:
            # 奖励曲线图
            plt.figure(figsize=(8, 5))
            plt.plot(range(len(mean_rewards)), mean_rewards, 'o-', 
                     color='#1f77b4', linewidth=2, markersize=8)
            plt.xlabel('Sampling samples', fontsize=16)
            plt.ylabel('Normalized Reward', fontsize=16)
            plt.title('Reward Dynamics in Diffusion Process', fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{save_path}/step_reward_curve.png', dpi=400, bbox_inches='tight')
            plt.close()
            
            # 差值柱状图
            plt.figure(figsize=(8, 5))
            plt.bar(range(len(diffs)), diffs, 
                    color='#2874a6', alpha=0.8)
            plt.xlabel('Sampling Step', fontsize=16)
            plt.ylabel('ΔR (Normalized Reward)', fontsize=16)
            plt.title('Reward by Diffusion Step', fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{save_path}/reward_diff_histogram.png', dpi=400, bbox_inches='tight')
            plt.close()

    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    pipeline()

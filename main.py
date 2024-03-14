import argparse
from time import sleep
import gymnasium as gym
from mpo import MPO
import os
import json
from datetime import datetime
import shutil
from scipy.optimize import minimize
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.distributions import MultivariateNormal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from mpoNetworks import PolicyNet, QNet, ReplayBuffer
from tensorboardX import SummaryWriter
import cProfile
import stable_baselines3
from stable_baselines3 import SAC, PPO, A2C, DDPG

# Create directories to hold models and logs
model_dir = "saved_models"
log_dir = "training_logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

def get_algorithm_class(algo_name):
    """
    Attempt to get the RL algorithm class based on algo_name.
    Raises an exception if the algorithm is not found.
    """
    try:
        return getattr(stable_baselines3, algo_name)
    except AttributeError:
        raise ValueError(f"Algorithm {algo_name} not found. Please verify the name.")

def train_baseline(env, env_name, algo_name):
    try:
        AlgoClass = get_algorithm_class(algo_name)
        this_log_dir = log_dir + '/' + env_name
        model = AlgoClass('MlpPolicy', env, verbose=1, device='cpu', tensorboard_log=this_log_dir)
    except ValueError as e:
        print(e)
        return

    TIMESTEPS = 25000
    step = 0
    while True:
        step += 1

        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{model_dir}/{env_name}/{algo_name}/{algo_name}_{TIMESTEPS*step}")

def test_baseline(env, algo_name, path_to_model):
    try:
        AlgoClass = get_algorithm_class(algo_name)
        model = AlgoClass.load(path_to_model, env=env)
    except ValueError as e:
        print(e)
        return

    obs = env.reset()[0]
    done = False
    extra_steps = 500
    while True:
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)

        if done:
            extra_steps -= 1

            if extra_steps < 0:
                break


def main():
    # Set up the parser
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                    description='Train and evaluate an RL model using MPO or other algorithms on OpenAI Gym environments.')

    # Environment and device configuration
    parser.add_argument('--env', type=str, default='HalfCheetah-v4', help='Name of the OpenAI Gym environment.')
    parser.add_argument('--device', type=str, default='cpu', help='Compute device to use (e.g., "cpu", "cuda").')

    # Algorithm selection
    parser.add_argument('--algo_name', type=str, default='MPO', help='RL algorithm to use (e.g., MPO, SAC, PPO, A2C).')

    # Hyperparameters for the chosen algorithm
    parser.add_argument('--dual_constraint', type=float, default=0.01, help='KL divergence dual constraint in E-step.')
    parser.add_argument('--kl_mean_constraint', type=float, default=0.01, help='KL divergence mean constraint in M-step.')
    parser.add_argument('--kl_var_constraint', type=float, default=0.0001, help='KL divergence covariance constraint in M-step.')
    parser.add_argument('--discount_factor', type=float, default=0.99, help='Discount factor for future rewards.')
    parser.add_argument('--adam_learning_rate', type=float, default=0.0005, help='Learning rate for Adam optimizer.')

    # Network architecture
    parser.add_argument('--policyNet_size', type=int, default=100, help='Size (neurons per layer) of the policy network.')
    parser.add_argument('--qNet_size', type=int, default=200, help='Size (neurons per layer) of the Q-network.')

    # Training configuration
    parser.add_argument('--sample_episode_num', type=int, default=50, help='Number of episodes to sample per training iteration.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--sample_action_num', type=int, default=64, help='Number of actions to sample during training.')
    parser.add_argument('--iteration_num', type=int, default=1000, help='Total number of training iterations.')
    parser.add_argument('--episode_rerun_num', type=int, default=3, help='Number of times to rerun sampled episodes for training.')

    # Evaluation configuration
    parser.add_argument('--evaluate_period', type=int, default=10, help='Number of iterations between evaluations.')
    parser.add_argument('--evaluate_episode_num', type=int, default=10, help='Number of episodes to use for each evaluation.')
    parser.add_argument('--evaluate_episode_maxstep', type=int, default=300, help='Maximum number of steps per evaluation episode.')

    # Loss and optimization
    parser.add_argument('--q_loss_type', type=str, default='mse', help='Type of loss function for Q-value optimization (e.g., "mse").')
    parser.add_argument('--alpha_mean_scale', type=float, default=1.0, help='Scaling factor for the Lagrangian multiplier related to mean in M-step.')
    parser.add_argument('--alpha_var_scale', type=float, default=100.0, help='Scaling factor for the Lagrangian multiplier related to variance in M-step.')
    parser.add_argument('--alpha_mean_max', type=float, default=0.1, help='Maximum value for the Lagrangian multiplier related to mean.')
    parser.add_argument('--alpha_var_max', type=float, default=10.0, help='Maximum value for the Lagrangian multiplier related to variance.')
    parser.add_argument('--sample_episode_maxstep', type=int, default=300, help='maximum length of a sampled episode')
    parser.add_argument('--mstep_iteration_num', type=int, default=10, help='Number of iterations for the M-Step optimization.')

    # Logging and rendering
    parser.add_argument('--log_dir', type=str, default="training_logs/", help='Directory for training logs.')
    parser.add_argument('--model_dir', type=str, default="saved_models/", help='Directory for saving trained models.')
    parser.add_argument('--load', type=str, default=None, help='load path')
    parser.add_argument('--overwrite_logs', type=bool, default=False, help='Whether to overwrite existing logs.')
    parser.add_argument('--render', type=bool, default=False, help='Whether to render the environment for visualization.')
    parser.add_argument('--model_name', type=str, default="MPO_/model_latest.pt", help='Path to a specific model for evaluation or further training.')

    args = parser.parse_args()

    # Setup environment with the appropriate render mode
    render_mode = 'human' if args.render else None
    env = gym.make(args.env, render_mode=render_mode)

    if args.algo_name == 'MPO':
        # Configure directories for MPO
        args.log_dir += f'{args.env}/MPO_'
        args.model_dir += f'{args.env}/MPO_'

        # Initialize and potentially load MPO model
        model = MPO(env, args)
        if args.load:
            model.load_model(args.load)
        
        if args.render:
            model_path = f"saved_models/{args.env}/{args.model_name}"
            model.load_model(model_path)
            
            state = env.reset(seed=123)[0] # Initialize state with a fixed seed
            for _ in range(1000):  # Max steps to render
                action = model.policy.action(torch.from_numpy(state).type(torch.float32).to(model.device)).cpu().numpy()        
                state, _, terminated, truncated, _ = env.step(action)
                env.render()
                if terminated or truncated:
                    state = env.reset(seed=123)[0]
            env.close()
            return  # Exit after rendering
        else:
            # Handle directory management
            if os.path.exists(args.log_dir) or os.path.exists(args.model_dir):
                if args.overwrite_logs:
                    shutil.rmtree(args.log_dir, ignore_errors=True)
                    shutil.rmtree(args.model_dir, ignore_errors=True)
                    os.makedirs(args.log_dir)
                    os.makedirs(args.model_dir)
                    print(f"Previous logs and model directories deleted, new directories created at '{args.log_dir}' and '{args.model_dir}'.")
                else:
                    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
                    shutil.move(args.log_dir, f'{args.log_dir}{current_datetime}')
                    shutil.move(args.model_dir, f'{args.model_dir}{current_datetime}')
                    print(f"Existing directories have been renamed, and new directories are created at '{args.log_dir}' and '{args.model_dir}'.")
            os.makedirs(args.log_dir)
            os.makedirs(args.model_dir)

            # write hyperparameters
            with open(os.path.join(args.log_dir, 'hyperparametersMPO.txt'), 'a') as f:
                json.dump(args.__dict__, f, indent=2)
            model.learn(args.iteration_num)
    else:
        # For other algorithms
        if not args.render:
            train_baseline(env, args.env, args.algo_name)
        else:
            model_path = f"saved_models/{args.env}/{args.model_name}"
            test_baseline(env, args.algo_name, path_to_model=model_path)

    env.close()

if __name__ == '__main__':
    main()

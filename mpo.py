import os
from time import sleep
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.distributions import MultivariateNormal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from mpoNetworks import PolicyNet, QNet, ReplayBuffer
from tensorboardX import SummaryWriter

class MPO(object):
    """
    MPO Algorithm Implementation.
    Maximizes a policy's expected return while maintaining a KL divergence constraint between new and old policies.
    """
    def __init__(self, env, args):
        # Environment and hyperparameter setup
        self.env = env
        self.env_name = args.env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.device = args.device
        # KL divergence and optimization parameters
        self.eps_dual = args.dual_constraint
        self.eps_mu = args.kl_mean_constraint
        self.eps_sigma = args.kl_var_constraint
        self.gamma = args.discount_factor
        # Scaling and bounds for Lagrange multipliers
        self.alpha_mu_scale = args.alpha_mean_scale
        self.alpha_sigma_scale = args.alpha_var_scale
        self.alpha_mu_max = args.alpha_mean_max
        self.alpha_sigma_max = args.alpha_var_max
        # Training and evaluation settings
        self.sample_episode_num = args.sample_episode_num
        self.sample_episode_maxstep = args.sample_episode_maxstep
        self.sample_action_num = args.sample_action_num
        self.batch_size = args.batch_size
        self.episode_rerun_num = args.episode_rerun_num
        self.mstep_iteration_num = args.mstep_iteration_num
        self.evaluate_period = args.evaluate_period
        self.evaluate_episode_num = args.evaluate_episode_num
        self.evaluate_episode_maxstep = args.evaluate_episode_maxstep
        self.adam_learning_rate = args.adam_learning_rate
        # Neural network architecture
        self.policyNet_size = args.policyNet_size
        self.qNet_size = args.qNet_size
        # Policy and Q-networks
        self.policy = PolicyNet(env, self.policyNet_size).to(self.device)
        self.qnet = QNet(env, self.qNet_size).to(self.device)
        # Target networks for stable learning
        self.target_policy = PolicyNet(env, self.policyNet_size).to(self.device)
        self.target_qnet = QNet(env, self.qNet_size).to(self.device)
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.adam_learning_rate)
        self.qnet_optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.adam_learning_rate)
        # Loss function
        self.norm_loss_q = nn.MSELoss() if args.q_loss_type == 'mse' else nn.SmoothL1Loss()
        # Replay buffer for experience replay
        self.replaybuffer = ReplayBuffer()
        # Lagrangian multipliers for dual problem
        self.eta = np.random.rand()
        self.eta_mu = np.random.rand()
        self.eta_sigma = np.random.rand()
        # Logging and rendering setup
        self.max_return_eval = -np.inf
        self.start_iteration = 1
        self.render = False
        self.log_dir = args.log_dir
        self.model_dir = args.model_dir

    def __sample_trajectory_worker(self, i):
        """
        Worker function to sample a single trajectory using the target policy.
        """
        buff = []
        state, _ = self.env.reset()
        for _ in range(self.sample_episode_maxstep):
            action = self.target_policy.action(torch.from_numpy(state).type(torch.float32).to(self.device)).cpu().numpy()
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            buff.append((state, action, next_state, reward))
            if terminated or truncated:
                break
            else:
                state = next_state
        return buff

    def sample_trajectory(self, sample_episode_num):
        """
        Samples multiple trajectories using the __sample_trajectory_worker function.
        """
        self.replaybuffer.clear()
        episodes = [self.__sample_trajectory_worker(i)
                    for i in tqdm(range(sample_episode_num), desc='sampling trajectories')]
        self.replaybuffer.store_episodes(episodes)

    def learn(self, iteration_num=1000, model_save_period=100):
        """
        Main learning loop. Alternates between sampling trajectories, updating Q-function, and optimizing policy.
        """
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        writer = SummaryWriter(self.log_dir)

        # Main training loop
        for it in range(self.start_iteration, iteration_num + 1):
            self.sample_trajectory(self.sample_episode_num)
            buffer_size = len(self.replaybuffer)

            # Compute statistics for logging
            mean_reward = self.replaybuffer.mean_reward()
            mean_return = self.replaybuffer.mean_return()
            mean_episode_len = self.replaybuffer.mean_episode_len()
            mean_loss_q = []
            mean_loss_p = []
            mean_loss_l = []
            mean_est_q = []

            # Iterate over batches of experience
            for r in range(self.episode_rerun_num):
                for indices in tqdm(
                        BatchSampler(SubsetRandomSampler(range(buffer_size)), self.batch_size, drop_last=True),
                        desc='training {}/{}'.format(r+1, self.episode_rerun_num)):

                    # Extract batch for training
                    state_batch, action_batch, next_state_batch, reward_batch = zip(*[self.replaybuffer[index] for index in indices])
                    state_batch = torch.from_numpy(np.stack(state_batch)).type(torch.float32).to(self.device)
                    action_batch = torch.from_numpy(np.stack(action_batch)).type(torch.float32).to(self.device)
                    next_state_batch = torch.from_numpy(np.stack(next_state_batch)).type(torch.float32).to(self.device)
                    reward_batch = torch.from_numpy(np.stack(reward_batch)).type(torch.float32).to(self.device)

                    # Update Q-function
                    loss_q, q = self.qnet_update_td(state_batch, action_batch, next_state_batch, reward_batch, self.sample_action_num)
                    mean_loss_q.append(loss_q.item())
                    mean_est_q.append(q.abs().mean().item())

                    # E-step: Estimate policy update
                    with torch.no_grad():
                        b_mu, b_A = self.target_policy.forward(state_batch)
                        b = MultivariateNormal(b_mu, scale_tril=b_A)
                        sampled_actions = b.sample((N,))
                        expanded_states = state_batch[None, ...].expand(N, -1, -1)
                        target_q = self.target_qnet.forward(expanded_states.reshape(-1, self.state_dim), sampled_actions.reshape(-1, self.action_dim)).reshape(N, K)
                        target_q_np = target_q.cpu().transpose(0, 1).numpy()

                    # Solve dual problem
                    def dual(eta):
                        max_q = np.max(target_q_np / (eta), axis=1, keepdims=True)
                        stable_log_exp_mean = max_q + np.log(np.mean(np.exp(target_q_np / eta - max_q), axis=1))
                        return eta * self.eps_dual + eta * np.mean(stable_log_exp_mean)

                    res = minimize(dual, np.array([self.eta]), method='SLSQP', bounds=[(1e-6, None)])
                    self.eta = res.x[0]

                    # Normalize target Q-values
                    norm_target_q = torch.softmax(target_q / self.eta, dim=0)

                    # M-step: Optimize policy
                    for _ in range(self.mstep_iteration_num):
                        mu, A = self.policy.forward(state_batch)
                        policy = MultivariateNormal(loc=mu, scale_tril=A)
                        loss_p = torch.mean(norm_target_q * policy.expand((N, K)).log_prob(sampled_actions))
                        C_mu, C_sigma = gaussian_kl(mu_i=b_mu, mu=mu, Ai=b_A, A=A)

                        mean_loss_p.append((-loss_p).item())

                        # Update Lagrange multipliers
                        self.eta_mu -= self.alpha_mu_scale * (self.eps_mu - C_mu).detach().item()
                        self.eta_sigma -= self.alpha_sigma_scale * (self.eps_sigma - C_sigma).detach().item()
                        self.eta_mu = np.clip(self.eta_mu, 0.0, self.alpha_mu_max)
                        self.eta_sigma = np.clip(self.eta_sigma, 0.0, self.alpha_sigma_max)

                        # Gradient step
                        self.policy_optimizer.zero_grad()
                        loss_l = -(loss_p + self.eta_mu * (self.eps_mu - C_mu) + self.eta_sigma * (self.eps_sigma - C_sigma))
                        mean_loss_l.append(loss_l.item())
                        loss_l.backward()
                        clip_grad_norm_(self.policy.parameters(), 0.1)
                        self.policy_optimizer.step()

            # Update target networks
            self.update_target_nets()

            # Save model and log training progress
            self.save_model(it, os.path.join(self.model_dir, 'model_latest.pt'))
            if it % model_save_period == 0:
                self.save_model(it, os.path.join(self.model_dir, 'model_{}.pt'.format(it)))

            # Log training progress
            mean_loss_q = np.mean(mean_loss_q)
            mean_loss_p = np.mean(mean_loss_p)
            mean_loss_l = np.mean(mean_loss_l)
            mean_est_q = np.mean(mean_est_q)

            print('iteration :', it)
            if it % self.evaluate_period == 1:
                self.policy.eval()
                return_eval = self.evaluate()
                self.policy.train()
                self.max_return_eval = max(self.max_return_eval, return_eval)
                print('  max_return_eval :', self.max_return_eval)
                print('  return_eval :', return_eval)
                writer.add_scalar('mpo_data/max_return_eval', self.max_return_eval, it)
                writer.add_scalar('mpo_data/return_eval', return_eval, it)
            print('  mean return :', mean_return)
            print('  mean episode length :', mean_episode_len)
            print('  mean loss_q :', mean_loss_q)
            print('  eta :', self.eta)
            writer.add_scalar('rollout/ep_rew_mean', mean_return, it)
            writer.add_scalar('mpo_data/mean_reward', mean_reward, it)
            writer.add_scalar('mpo_data/loss_q', mean_loss_q, it)
            writer.add_scalar('mpo_data/loss_p', mean_loss_p, it)
            writer.add_scalar('mpo_data/loss_l', mean_loss_l, it)
            writer.add_scalar('mpo_data/mean_q', mean_est_q, it)
            writer.add_scalar('mpo_data/eta', self.eta, it)
            writer.add_scalar('mpo_data/eta_mu', self.eta_mu, it)
            writer.add_scalar('mpo_data/eta_sigma', self.eta_sigma, it)

            writer.flush()

        # Clean up after training
        if writer is not None:
            writer.close()

    def qnet_update_td(self, state_batch, action_batch, next_state_batch, reward_batch, sample_num=64):
        """
        Updates the Q-network by calculating the TD error.
        """
        B = state_batch.size(0)
        with torch.no_grad():
            pi_mean, pi_A = self.target_policy.forward(next_state_batch)  # Target policy prediction
            policy = MultivariateNormal(pi_mean, scale_tril=pi_A)  # Define normal distribution based on target policy
            sampled_next_actions = policy.sample((sample_num,)).transpose(0, 1)  # Sample actions from the policy
            expanded_next_states = next_state_batch[:, None, :].expand(-1, sample_num, -1)  # Expand state batch for each sampled action
            
            # Calculate the expected Q-value for next states and sampled actions
            expected_next_q = self.target_qnet.forward(
                expanded_next_states.reshape(-1, self.state_dim), 
                sampled_next_actions.reshape(-1, self.action_dim)
            ).reshape(B, sample_num).mean(dim=1)  # Take the mean across all samples
            
            y = reward_batch + self.gamma * expected_next_q  # Calculate target values
        self.qnet_optimizer.zero_grad()  # Reset gradients
        t = self.qnet(state_batch, action_batch).squeeze()  # Predict current Q-values
        loss = self.norm_loss_q(y, t)  # Calculate loss
        loss.backward()  # Backpropagation
        self.qnet_optimizer.step()  # Update weights
        return loss, y

    def update_target_nets(self):
        """
        Soft update the target networks.
        """
        # Update target policy network
        for target_param, param in zip(self.target_policy.parameters(), self.policy.parameters()):
            target_param.data.copy_(param.data)

        # Update target Q-network
        for target_param, param in zip(self.target_qnet.parameters(), self.qnet.parameters()):
            target_param.data.copy_(param.data)

    def load_model(self, path=None):
        """
        Loads model weights from a specified path.
        """
        load_path = path if path is not None else self.save_path
        checkpoint = torch.load(load_path)
        self.start_iteration = checkpoint['iteration'] + 1
        self.qnet.load_state_dict(checkpoint['qnet_state_dict'])
        self.target_qnet.load_state_dict(checkpoint['target_qnet_state_dict'])
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.target_policy.load_state_dict(checkpoint['target_policy_state_dict'])
        self.qnet_optimizer.load_state_dict(checkpoint['qnet_optim_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optim_state_dict'])
        self.qnet.train()
        self.target_qnet.train()
        self.policy.train()
        self.target_policy.train()

    def save_model(self, it, path=None):
        """
        Saves model weights to a specified path.
        """
        data = {
            'iteration': it,
            'policy_state_dict': self.policy.state_dict(),
            'target_policy_state_dict': self.target_policy.state_dict(),
            'qnet_state_dict': self.qnet.state_dict(),
            'target_qnet_state_dict': self.target_qnet.state_dict(),
            'policy_optim_state_dict': self.policy_optimizer.state_dict(),
            'qnet_optim_state_dict': self.qnet_optimizer.state_dict()
        }
        torch.save(data, path)

    def evaluate(self):
        """
        Evaluates the current policy by running it in the environment.
        """
        with torch.no_grad():
            total_rewards = []
            for e in tqdm(range(self.evaluate_episode_num), desc='evaluating'):
                total_reward = 0.0
                state, _ = self.env.reset()
                for s in range(self.evaluate_episode_maxstep):
                    action = self.policy.action(torch.from_numpy(state).type(torch.float32).to(self.device)).cpu().numpy()
                    state, reward, terminated, truncated, _ = self.env.step(action)
                    total_reward += reward
                    if terminated or truncated:
                        break
                total_rewards.append(total_reward)
            return np.mean(total_rewards)

def bt(m): # Get transpose of a matrix.
    return m.transpose(dim0=-2, dim1=-1)

def btr(m): # Get trace of a matrix.
    return m.diagonal(dim1=-2, dim2=-1).sum(-1)

def gaussian_kl(mu_i, mu, Ai, A):
    """
    Computes the decoupled KL divergence between two multivariate Gaussian distributions.
    """
    n = A.size(-1)
    mu_i = mu_i.unsqueeze(-1)  # Reshape for batch matrix multiplication
    mu = mu.unsqueeze(-1)
    sigma_i = Ai @ bt(Ai)  # Compute covariance matrix for distribution i
    sigma = A @ bt(A)  # Compute covariance matrix for distribution
    sigma_i_det = sigma_i.det()  # Compute determinant of covariance matrices
    sigma_det = sigma.det()
    sigma_i_det = torch.clamp_min(sigma_i_det, 1e-6)  # Avoid numerical issues
    sigma_det = torch.clamp_min(sigma_det, 1e-6)
    sigma_i_inv = sigma_i.inverse()  # Compute inverse of covariance matrices
    sigma_inv = sigma.inverse()

    inner_mu = ((mu - mu_i).transpose(-2, -1) @ sigma_i_inv @ (mu - mu_i)).squeeze()  # Compute inner term for mu
    inner_sigma = torch.log(sigma_det / sigma_i_det) - n + btr(sigma_inv @ sigma_i)  # Compute inner term for sigma
    C_mu = 0.5 * torch.mean(inner_mu)  # Mean term of the KL divergence
    C_sigma = 0.5 * torch.mean(inner_sigma)  # Covariance term of the KL divergence
    return C_mu, C_sigma

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import MultivariateNormal


class QNet(nn.Module):
    """
    Q network
    :param env: Gymnasium environment to determine input size
    :param qNet_size: number of neurons in each layer (2 fully connected hidden layers in total)
    """
    def __init__(self, env, qNet_size):
        super(QNet, self).__init__()
        self.env = env
        self.nStates = env.observation_space.shape[0]
        self.nActions = env.action_space.shape[0]
        self.lin1 = nn.Linear(self.nStates + self.nActions, qNet_size)
        self.lin2 = nn.Linear(qNet_size, qNet_size)
        self.lin3 = nn.Linear(qNet_size, 1)

    def forward(self, state, action):
        """
        :param state: (B, nStates); B = Batch size
        :param action: (B, nActions)
        :return: Q-value
        """
        h = torch.cat([state, action], dim=1)  # (B, nStates+nActions)
        h = F.relu(self.lin1(h))  # (B, n)
        h = F.relu(self.lin2(h))  # (B, n)
        v = self.lin3(h)  # (B, 1)
        return v


class PolicyNet(nn.Module):
    """
    Policy network
    :param env: OpenAI gym environment to determine input size
    :param policyNet_size: number of neurons in each layer (2 fully connected hidden layers in total)
    """
    def __init__(self, env, policyNet_size):
        super(PolicyNet, self).__init__()
        self.env = env
        self.nStates = env.observation_space.shape[0]
        self.nActions = env.action_space.shape[0]
        self.lin1 = nn.Linear(self.nStates, policyNet_size)
        self.lin2 = nn.Linear(policyNet_size, policyNet_size)
        self.action_means = nn.Linear(policyNet_size, self.nActions)
        self.actions_covar = nn.Linear(policyNet_size, (self.nActions * (self.nActions + 1)) // 2)

    def forward(self, state):
        """
        :param state: (B, nStates)
        :return: mean values of all actions (B, nActions) and cholesky factorization of their covariance matrix (B, nActions*(nActions + 1)//2)
        """
        device = state.device
        B = state.size(0)
        nStates = self.nStates
        nActions = self.nActions
        action_low = torch.from_numpy(self.env.action_space.low)[None, ...].to(device)  # (1, nActions)
        action_high = torch.from_numpy(self.env.action_space.high)[None, ...].to(device)  # (1, nActions)
        x = F.relu(self.lin1(state))
        x = F.relu(self.lin2(x))
        mean = torch.sigmoid(self.action_means(x))  # (B, nActions) # Values from 0 and 1
        mean = action_low + (action_high - action_low) * mean # Scale the values between the lower and upper limits of the actions
        cholesky_vector = self.actions_covar(x)  # (B, (nActions*(nActions+1))/2)
        cholesky_diag_index = torch.arange(nActions, dtype=torch.long) + 1
        cholesky_diag_index = (cholesky_diag_index * (cholesky_diag_index + 1)) // 2 - 1
        cholesky_vector[:, cholesky_diag_index] = F.softplus(cholesky_vector[:, cholesky_diag_index])
        tril_indices = torch.tril_indices(row=nActions, col=nActions, offset=0)
        cholesky = torch.zeros(size=(B, nActions, nActions), dtype=torch.float32).to(device)
        cholesky[:, tril_indices[0], tril_indices[1]] = cholesky_vector
        return mean, cholesky

    def action(self, state):
        """
        :param state: (nStates,)
        :return: an action
        """
        with torch.no_grad():
            mean, cholesky = self.forward(state[None, ...])
            action_distribution = MultivariateNormal(mean, scale_tril=cholesky)
            action = action_distribution.sample()
        return action[0]


class ReplayBuffer:
    def __init__(self):

        # buffers
        self.start_idx_of_episode = []
        self.idx_to_episode_idx = []
        self.episodes = []
        self.tmp_episode_buff = []

    def store_step(self, state, action, next_state, reward):
        self.tmp_episode_buff.append((state, action, next_state, reward))

    def store_episodes(self, episodes):
        for episode in episodes:
            states, actions, next_states, rewards = zip(*episode)
            episode_len = len(states)
            usable_episode_len = episode_len - 1
            self.start_idx_of_episode.append(len(self.idx_to_episode_idx))
            self.idx_to_episode_idx.extend([len(self.episodes)] * usable_episode_len)
            self.episodes.append((states, actions, next_states, rewards))

    def clear(self):
        self.start_idx_of_episode = []
        self.idx_to_episode_idx = []
        self.episodes = []
        self.tmp_episode_buff = []

    def __getitem__(self, idx):
        episode_idx = self.idx_to_episode_idx[idx]
        start_idx = self.start_idx_of_episode[episode_idx]
        i = idx - start_idx
        states, actions, next_states, rewards = self.episodes[episode_idx]
        state, action, next_state, reward = states[i], actions[i], next_states[i], rewards[i]
        return state, action, next_state, reward

    def __len__(self):
        return len(self.idx_to_episode_idx)

    def mean_reward(self):
        _, _, _, rewards = zip(*self.episodes)
        return np.mean([np.mean(reward) for reward in rewards])

    def mean_return(self):
        _, _, _, rewards = zip(*self.episodes)
        return np.mean([np.sum(reward) for reward in rewards])

    def mean_episode_len(self):
        states, _, _, _ = zip(*self.episodes)
        return np.mean([np.mean(len(state)) for state in states])

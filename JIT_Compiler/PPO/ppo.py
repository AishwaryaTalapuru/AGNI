import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
import os


# Defining the PPO Agent
class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_epsilon=0.2, batch_size=64, epochs=10):
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        self.optimizer = optim.Adam(list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.batch_size = batch_size
        self.epochs = epochs
        self.memory = deque(maxlen=10000)
    
    def act(self, state):
        state = torch.FloatTensor(state)
        action_probs = self.policy_net(state)
        action = torch.multinomial(action_probs, 1).item()
        return action
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def compute_returns(self, rewards, dones):
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        return torch.FloatTensor(returns)
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sampling a batch from mem
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Computing returns and advantages
        returns = self.compute_returns(rewards, dones)
        values = self.value_net(states)
        advantages = returns - values.detach()
        
        # Training for multiple epochs
        for _ in range(self.epochs):
            # Computing new action probabilities and values
            new_action_probs = self.policy_net(states)
            new_values = self.value_net(states)
            
            # Computing the ratio of new and old action probabilities
            old_action_probs = new_action_probs.detach()
            ratio = new_action_probs.gather(1, actions.unsqueeze(1)) / old_action_probs.gather(1, actions.unsqueeze(1))
            
            # Computing the clipped surrogate objective
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            #surrogate2 = ratio
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            
            # Computing the value loss
            value_loss = F.mse_loss(new_values.squeeze(), returns)
            
            # Comput entropy bonus (optional)
            entropy = -(new_action_probs * torch.log(new_action_probs + 1e-10)).sum(dim=1).mean()
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            
            # Updating the networks
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def save_models(self, path="models"):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.policy_net.state_dict(), os.path.join(path, "policy_net.pth"))
        torch.save(self.value_net.state_dict(), os.path.join(path, "value_net.pth"))
        print("Models saved successfully.")
    
    def load_models(self, path="models"):
        if os.path.exists(os.path.join(path, "policy_net.pth")):
            self.policy_net.load_state_dict(torch.load(os.path.join(path, "policy_net.pth")))
            self.value_net.load_state_dict(torch.load(os.path.join(path, "value_net.pth")))
            print("Models loaded successfully.")
        else:
            print("No saved models found.")


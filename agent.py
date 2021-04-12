import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from torch.distributions import Categorical, Normal
from itertools import count
from tqdm import tqdm

from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic_OneStep():
    def __init__(self, env, actor_model, critic_model, actor_lr, critic_lr, gamma):
        self.env = env
        self.actor = actor_model.to(device)
        self.critic = critic_model.to(device)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.optim_actor = optim.Adam(self.actor.parameters(), lr = self.actor_lr)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr = self.critic_lr)
        self.gamma = gamma
  
    def update(self, log_prob, delta):
        critic_loss = (delta)**2
        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.optim_critic.step()
        
        actor_loss = -log_prob * delta.detach()
        self.optim_actor.zero_grad()
        actor_loss.backward()
        self.optim_actor.step()

    def learn_task(self, n_episodes):
        episode_rewards = []
        avg_rewards = []

        for episode in tqdm( range(n_episodes) ):
            state = self.env.reset()
            rewards = []
            state = torch.FloatTensor(state).to(device)

            for i in count():
                dist, value = self.actor(state), self.critic(state)
                action = dist.sample()
                next_state, reward, done, _ = self.env.step(action.cpu().numpy())
                rewards.append(reward)

                next_state = torch.FloatTensor(next_state).to(device)
                next_value = self.critic(next_state)
                log_prob = dist.log_prob(action).unsqueeze(0)
                
                if done:
                    delta = reward+self.gamma*0-value
                    self.update(log_prob, delta)
                    print(sum(rewards))
                    episode_rewards.append(sum(rewards))
                    break
                    
                delta = reward+self.gamma*next_value-value
                self.update(log_prob, delta)
                state = next_state

            if episode >= 100:
                avg_rewards.append(np.mean(episode_rewards[-100:]))
        return episode_rewards, avg_rewards
    
    def learn_task_continuous(self, n_episodes):
        episode_rewards = []
        avg_rewards = []

        for episode in tqdm( range(n_episodes) ):
            state = self.env.reset()
            rewards = []
            state = torch.FloatTensor(state).to(device)
            for i in count():
                dist, value = self.actor(state), self.critic(state)
                probs = dist.sample()
                action = torch.tanh(probs).item()
                
                next_state, reward, done, _ = self.env.step(np.array(action).reshape((1,)))
                rewards.append(reward)
                next_state = torch.FloatTensor(next_state).to(device)
                next_value = self.critic(next_state)
                log_prob = dist.log_prob(probs).to(device)
                
                if done:
                    delta = reward+self.gamma*0-value
                    self.update(log_prob, delta)
                    print(sum(rewards))
                    episode_rewards.append(sum(rewards))
                    break
                    
                delta = reward+self.gamma*next_value-value
                self.update(log_prob, delta)
                state = next_state
                
            if episode >= 100:
                avg_rewards.append(np.mean(episode_rewards[-100:]))
        return episode_rewards, avg_rewards
    
class ActorCritic_Batch():
    def __init__(self, env, actor_model, critic_model, actor_lr, critic_lr, gamma):
        self.env = env
        self.actor = actor_model.to(device)
        self.critic = critic_model.to(device)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.optim_actor = optim.Adam(self.actor.parameters(), lr = self.actor_lr)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr = self.critic_lr)
        self.gamma = gamma
  
    def update(self, actor_loss, critic_loss):
        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.optim_critic.step()
        
        self.optim_actor.zero_grad()
        actor_loss.backward()
        self.optim_actor.step()

    def compute_returns(self, rewards, masks):
        returns = []
        for t in range(len(rewards)):
            _return = 0
            for t_i in range(len(masks[t:])):
                _return += masks[t+t_i]*rewards[t+t_i]
            returns.append(_return)
        returns = torch.cat(returns).detach()
        return returns
    
    def render_episode(self):
        state = self.env.reset()
        state = torch.FloatTensor(state).to(device)
        
        score = 0
        while True:
            self.env.render()
            dist = self.actor(state)
            action = dist.sample()
                
            next_state, reward, done, _ = self.env.step(action.cpu().numpy())
            next_state = torch.FloatTensor(next_state).to(device)
            
            state = next_state
            score += reward
            
            if done:
                print(score)
                break
    
    def learn_task(self, n_episodes):
        episode_rewards = []
        avg_rewards = []

        for episode in tqdm( range(n_episodes) ):
            state = self.env.reset()
            state = torch.FloatTensor(state).to(device)

            log_probs = []
            values = []
            rewards = []
            masks = []

            for i in count():
                dist, value = self.actor(state), self.critic(state)
                action = dist.sample()
                
                next_state, reward, done, _ = self.env.step(action.cpu().numpy())

                next_state = torch.FloatTensor(next_state).to(device)
                next_value = self.critic(next_state)
                log_prob = dist.log_prob(action).unsqueeze(0)

                rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
                values.append(value)
                log_probs.append(log_prob)
                masks.append(torch.tensor([self.gamma**i], dtype=torch.float, device=device))

                if done:
                    print(sum(rewards).item())
                    episode_rewards.append(sum(rewards).item())
                    break

                state = next_state

            returns = self.compute_returns(rewards, masks)
            values = torch.cat(values).to(device)
            log_probs = torch.stack(log_probs).to(device)
                        
            deltas = returns - values
            
            critic_loss = torch.mean(deltas**2)
            actor_loss = torch.mean(-log_probs * deltas.detach())
            
            self.update(actor_loss, critic_loss)

            if episode >= 100:
                avg_rewards.append(np.mean(episode_rewards[-100:]))
        return episode_rewards, avg_rewards
    
    def learn_task_continuous(self, n_episodes):
        episode_rewards = []
        avg_rewards = []

        for episode in tqdm( range(n_episodes) ):
            state = self.env.reset()
            state = torch.FloatTensor(state).to(device)

            log_probs = []
            values = []
            rewards = []
            masks = []

            for i in count():
                dist, value = self.actor(state), self.critic(state)
                probs = dist.sample()
                action = torch.tanh(probs).item()
                next_state, reward, done, _ = self.env.step(np.array(action).reshape((1,)))

                next_state = torch.FloatTensor(next_state).to(device)
                next_value = self.critic(next_state)
                log_prob = dist.log_prob(probs).to(device)

                rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
                values.append(value)
                log_probs.append(log_prob)
                masks.append(torch.tensor([self.gamma**i], dtype=torch.float, device=device))

                if done:
                    print(sum(rewards).item())
                    episode_rewards.append(sum(rewards).item())
                    break

                state = next_state
                
            returns = self.compute_returns(rewards, masks)
            values = torch.cat(values).to(device)
            log_probs = torch.stack(log_probs).to(device)
                        
            deltas = returns - values
            
            critic_loss = torch.mean(deltas**2)
            actor_loss = torch.mean(-log_probs * deltas.detach())
            
            self.update(actor_loss, critic_loss)

            if episode >= 100:
                avg_rewards.append(np.mean(episode_rewards[-100:]))
        return episode_rewards, avg_rewards
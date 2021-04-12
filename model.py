import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions import Categorical, Normal

class Actor(nn.Module):
    """
    Feel free to change the architecture for different tasks!
    """
    def __init__(self, input_size, output_size, hidden_size):
        super(Actor, self).__init__()
        self.state_size = input_size
        self.action_size = output_size
        self.linear1 = nn.Linear(self.state_size, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.linear3 = nn.Linear(hidden_size[1], self.action_size)
        
    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution

class Actor_Continuous(nn.Module):
    """
    Feel free to change the architecture for different tasks!
    """
    def __init__(self, input_size, hidden_size):
        super(Actor_Continuous, self).__init__()
        self.state_size = input_size
        self.linear1 = nn.Linear(self.state_size, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.linear3 = nn.Linear(hidden_size[1], 2) #mu and sigma
        
    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        
        distribution = torch.distributions.Normal(output[0], torch.exp(output[1]))
        return distribution

class Critic(nn.Module):
    """
    Feel free to change the architecture for different tasks!
    """
    def __init__(self, input_size, hidden_size):
        super(Critic, self).__init__()
        self.state_size = input_size
        self.linear1 = nn.Linear(self.state_size, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.linear3 = nn.Linear(hidden_size[1], 1) # Note the single value - this is V(s)!
        
    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value
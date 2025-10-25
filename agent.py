

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))  # run 2 adding one more layer 
        x = F.relu(self.fc3(x))
        return self.fc4(x)
    

Experience = namedtuple("Experience",
                         field_names=["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """Save an experience."""
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, state_size: int, n_actions: int,
                 buffer_size: int = 10000, batch_size: int = 64,   #64,
                 gamma: float = 0.99, lr: float = 1e-3, 
                 tau: float = 5e-4,        
                 update_every: int = 4,    
                 target_update_every: int = 100, 
                 eps_start: float = 1.0, eps_end: float = 0.01, eps_decay: float = 0.995):

        self.state_size = state_size
        self.n_actions = n_actions
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.tau = tau 
        self.update_every = update_every
        self.target_update_every = target_update_every 

        # Epsilon-greedy parameters
        self.eps = eps_start
        self.eps_min = eps_end
        self.eps_decay = eps_decay

        # --- D Q-Networks ---
        self.qnetwork_local = DQN(state_size, n_actions).float() 
        self.qnetwork_target = DQN(state_size, n_actions).float()
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict()) 
        self.qnetwork_target.eval() 


        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss() # Loss function

        self.memory = ReplayBuffer(self.buffer_size)

        self.t_step = 0 # For self.update_every
        self.target_t_step = 0 # For self.target_update_every

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.memory.push(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self.learn(experiences)

        self.target_t_step = (self.target_t_step + 1) % self.target_update_every
        if self.target_t_step == 0:
            self.hard_update_target_network()

    def choose_action(self, state):
        """ε-greedy action selection using the local Q-network."""
        if random.random() < self.eps:
            return random.choice(np.arange(self.n_actions))
        else:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0) 
            self.qnetwork_local.eval() 
            with torch.no_grad():
                action_values = self.qnetwork_local(state_tensor)
            self.qnetwork_local.train() 
            return np.argmax(action_values.cpu().data.numpy())


    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples."""
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.from_numpy(np.vstack(states)).float()
        actions = torch.from_numpy(np.vstack(actions)).long() 
        rewards = torch.from_numpy(np.vstack(rewards)).float()
        next_states = torch.from_numpy(np.vstack(next_states)).float()
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float() 

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = self.criterion(Q_expected, Q_targets)

        self.optimizer.zero_grad() 
        loss.backward()           
        self.optimizer.step()      


    def hard_update_target_network(self):
        """Hard update: Copy weights from local network to target network."""
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def decay_epsilon(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

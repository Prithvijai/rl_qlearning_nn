import numpy as np
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, n_actions: int, alpha: float = 0.5, gamma: float = 0.99,
                 eps_start: float = 0.3, eps_end: float = 0.01, eps_decay: float = 0.9995):
        """
        Q learning Agent which calculate Q table
        """

        self.actions = list(range(n_actions))
        self.alpha = float(alpha)
        self.gamma = float(gamma)

        self.eps = float(eps_start)
        self.eps_min = float(eps_end)
        self.eps_decay = float(eps_decay)

        # Q[(state_tuple, action)] -> float
        self.Q = defaultdict(float)

    def choose_action(self, state):
        """ε-greedy over actions with random tie-breaking."""
        if random.random() < self.eps:
            return random.choice(self.actions)
        qvals = [self.Q[(state, a)] for a in self.actions]
        maxv = max(qvals)
        best = [a for a, v in enumerate(qvals) if v == maxv]
        return random.choice(best)

    def update(self, state, action, reward, next_state, terminated: bool):
        """One-step Q-learning; only zero the bootstrap on TRUE termination."""
        q_sa = self.Q[(state, action)]
        next_max = 0.0 if terminated else max(self.Q[(next_state, a)] for a in self.actions)
        td_target = reward + self.gamma * next_max
        self.Q[(state, action)] = q_sa + self.alpha * (td_target - q_sa)

    def decay_epsilon(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame   

class GridWorldEnv(gym.Env):
    """
    3x4 grid, zero-indexed (row, col):
    Stochastic dynamics: 0.8 desired direction / 0.1 left of desired direction / 0.1 right of desired direction
    """
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 4}

    def __init__(self, render_mode=None, step_reward: float = -0.04, trap_reward: int = -1):
        super().__init__()

        self.rows, self.cols = 3, 4 
        self.goal_reward, self.trap_reward, self.step_reward = 1.0, trap_reward, step_reward

        self.observation_space = spaces.MultiDiscrete([self.rows, self.cols])
        self.action_space = spaces.Discrete(4)

        self.start_pos = [2, 0]
        self.goal_pos  = [0, 3]
        self.trap_pos  = [1, 3]
        self.wall_pos  = [1, 1]
        self.agent_pos = None

        self.render_mode = render_mode
        self.window_size = 512
        self.window, self.clock = None, None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array(self.start_pos, dtype=int)

        if self.render_mode == "human":
            self._render_frame()
        
        return self.agent_pos.copy(), {}

    def step(self, action: int):
        # Dynamic Env
        p = self.np_random.random()
        if p < 0.8:
            actual_action = action
        elif p < 0.9:
            actual_action = (action - 1) % 4
        else:
            actual_action = (action + 1) % 4

        prev_pos = self.agent_pos.copy()

        # Apply action: 0:UP, 1:RIGHT, 2:DOWN, 3:LEFT
        if actual_action == 0:
            self.agent_pos[0] -= 1
        elif actual_action == 1:
            self.agent_pos[1] += 1
        elif actual_action == 2:
            self.agent_pos[0] += 1
        elif actual_action == 3:
            self.agent_pos[1] -= 1

        # Keep inside bounds
        self.agent_pos[0] = np.clip(self.agent_pos[0], 0, self.rows - 1)
        self.agent_pos[1] = np.clip(self.agent_pos[1], 0, self.cols - 1)

        # Wall collision -> revert
        if np.array_equal(self.agent_pos, self.wall_pos):
            self.agent_pos = prev_pos

        # Rewarding 
        terminated = False
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward = self.goal_reward
            terminated = True
        elif np.array_equal(self.agent_pos, self.trap_pos):
            reward = self.trap_reward
            terminated = True
        else:
            reward = self.step_reward

        if self.render_mode == "human":
            self._render_frame()

        # return a copy of the state
        return self.agent_pos.copy(), float(reward), terminated, False, {}

    def render(self):
        if self.render_mode == "ansi":
            self._render_text()

    def _render_text(self):
        grid = np.full((self.rows, self.cols), '_', dtype=str)
        grid[tuple(self.start_pos)] = 'S'
        grid[tuple(self.goal_pos)]  = 'G'
        grid[tuple(self.trap_pos)]  = 'T'
        grid[tuple(self.wall_pos)]  = 'W'
        grid[tuple(self.agent_pos)] = 'A'
        for row in grid:
            print(' '.join(row))
        print("-" * (self.cols * 2 - 1))

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, int(self.window_size * self.rows / self.cols))
            )
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, int(self.window_size * self.rows / self.cols)))
        canvas.fill((255, 255, 255))
        pix = self.window_size / self.cols

        # Goal, Trap, Wall
        pygame.draw.rect(canvas, (0, 255, 0), pygame.Rect(self.goal_pos[1] * pix, self.goal_pos[0] * pix, pix, pix))
        pygame.draw.rect(canvas, (255, 0, 0), pygame.Rect(self.trap_pos[1] * pix, self.trap_pos[0] * pix, pix, pix))
        pygame.draw.rect(canvas, (0, 0, 0), pygame.Rect(self.wall_pos[1] * pix, self.wall_pos[0] * pix, pix, pix))

        # Agent
        center = ((self.agent_pos[1] + 0.5) * pix, (self.agent_pos[0] + 0.5) * pix)
        pygame.draw.circle(canvas, (0, 0, 255), center, pix / 3)

        # Grid lines
        h = int(self.window_size * self.rows / self.cols)
        for x in range(self.cols + 1):
            pygame.draw.line(canvas, (0, 0, 0), (x * pix, 0), (x * pix, h))
        for y in range(self.rows + 1):
            pygame.draw.line(canvas, (0, 0, 0), (0, y * pix), (self.window_size, y * pix))

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

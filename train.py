import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from env import GridWorldEnv
from agent import DQNAgent



ARROWS = {0: '↑', 1: '→', 2: '↓', 3: '←'}

def plot_rewards(scores, filename='dqn_learning_curve.png'):
    """Generates and saves a plot of rewards over episodes."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    
    # Calculate a moving average
    window_size = 100
    moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
    plt.plot(np.arange(len(moving_avg)) + window_size -1 , moving_avg, label=f'Moving Avg ({window_size} episodes)', color='red')

    plt.ylabel('Score (Total Reward per Episode)')
    plt.xlabel('Episode #')
    plt.title('DQN Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()
    print(f"Learning curve saved to {filename}")

def print_policy_dqn(agent, env):
    """Prints the final greedy policy derived from the DQN agent."""
    print("\n--- Final Policy (from DQN) ---")
    # Set the network to evaluation mode for inference
    agent.qnetwork_local.eval()
    
    with torch.no_grad():
        for r in range(env.rows):
            row_str = []
            for c in range(env.cols):
                s_np = np.array([r, c])
                s_tuple = (r, c)

                if s_tuple == tuple(env.goal_pos):
                    row_str.append('G')
                elif s_tuple == tuple(env.trap_pos):
                    row_str.append('T')
                elif s_tuple == tuple(env.wall_pos):
                    row_str.append('W')
                else:
                    # Convert state to a tensor, add a batch dimension, and get Q-values
                    state_tensor = torch.from_numpy(s_np).float().unsqueeze(0)
                    action_values = agent.qnetwork_local(state_tensor)
                    
                    # Choose the best action based on the highest Q-value
                    if torch.max(torch.abs(action_values)) > 1e-9: # Check if Q-values are non-trivial
                         action = np.argmax(action_values.cpu().data.numpy())
                         row_str.append(ARROWS[action])
                    else:
                         row_str.append('.') # If all Q-values are near zero
            print(' '.join(row_str))
            
    # Set the network back to training mode
    agent.qnetwork_local.train()


if __name__ == "__main__":
    
    env = GridWorldEnv(step_reward=-0.04)

    agent = DQNAgent(
        state_size=2,  # [row, col]
        n_actions=env.action_space.n,
        buffer_size=10000,
        batch_size=64,   # run 2 i will change the hidden layer from 2 to 3
        gamma=0.99,
        lr=1e-4,      #1e-3,        #5e-4,       # experiement 2 tuning learning rate 
        update_every=4,
        target_update_every=100, 
        eps_start=1.0, 
        eps_end=0.01,
        eps_decay=0.995     # experiment 3 change the decay rate
    )

    num_episodes = 2000
    max_steps_per_episode = 200
    
    all_scores = []
    scores_window = deque(maxlen=100) 

    print("--- Starting Training ---")
    for ep in range(1, num_episodes + 1):
        state_np, _ = env.reset()
        episode_reward = 0.0

        for t in range(max_steps_per_episode):
            action = agent.choose_action(state_np)
            next_state_np, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.step(state_np, action, reward, next_state_np, done)
            state_np = next_state_np
            episode_reward += reward
            
            if done:
                break

        scores_window.append(episode_reward)
        all_scores.append(episode_reward)
        agent.decay_epsilon()

        if ep % 100 == 0:
            avg_reward = np.mean(scores_window)
            print(f"\rEpisode {ep}/{num_episodes} | Avg Reward (last 100): {avg_reward:.2f} | ε: {agent.eps:.3f}")

    print("\n--- Training Finished ---")

    print_policy_dqn(agent, env)

    model_save_path = 'dqn_weights.pth'
    torch.save(agent.qnetwork_local.state_dict(), model_save_path)
    print(f"\nModel weights saved to {model_save_path}")

    plot_rewards(all_scores)

    print("\n--- Testing Final Policy (20 episodes, no exploration) ---")
    agent.eps = 0.0 
    successes = 0
    agent.qnetwork_local.eval() 

    for i in range(20):
        state_np, _ = env.reset()
        total_reward = 0.0
        
        for t in range(100): 
            state_tensor = torch.from_numpy(state_np).float().unsqueeze(0)
            with torch.no_grad():
                action_values = agent.qnetwork_local(state_tensor)
            action = np.argmax(action_values.cpu().data.numpy())

            next_state_np, r, term, _, _ = env.step(action)
            total_reward += r
            state_np = next_state_np
            
            if term:
                break
        
        final_pos = tuple(int(x) for x in state_np)
        if final_pos == tuple(env.goal_pos):
            outcome = "SUCCESS"
            successes += 1
        elif final_pos == tuple(env.trap_pos):
            outcome = "TRAP"
        else:
            outcome = "TIMEOUT"
            
        print(f"Test {i+1:02d}: {outcome:<8} | Total Reward: {total_reward:+.3f}")

    print(f"\nFinal Test Results: {successes}/20 successes ({successes/20*100:.1f}%)")
    env.close()
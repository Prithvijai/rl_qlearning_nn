import numpy as np
from env import GridWorldEnv
from agent import QLearningAgent
import json

ARROWS = {0: '↑', 1: '→', 2: '↓', 3: '←'}

def print_policy(Q, env):
    print("\n--- Final Policy ---")
    for r in range(env.rows):
        row = []
        for c in range(env.cols):
            s = (r, c)
            if s == tuple(env.goal_pos):
                row.append('G')
            elif s == tuple(env.trap_pos):
                row.append('T')
            elif s == tuple(env.wall_pos):
                row.append('W')
            else:
                # best greedy action from this state
                qvals = [Q[(s, a)] for a in range(env.action_space.n)]
                if any(abs(v) > 1e-9 for v in qvals):
                    best = np.flatnonzero(qvals == np.max(qvals))
                    a = int(np.random.choice(best))
                    row.append(ARROWS[a])
                else:
                    row.append('.')
        print(' '.join(row))

def print_q_table(Q, env):
    print("\n--- Final Q-Table (state -> [U, R, D, L]) ---")
    seen_states = set(s for (s, _) in Q.keys())
    for s in sorted(seen_states):
        q = [Q[(s, a)] for a in range(env.action_space.n)]
        if any(abs(v) > 1e-12 for v in q):
            print(f"  {s}: {[f'{v: .3f}' for v in q]}")

def print_success_stats(recent_rewards):
    if not recent_rewards: return
    successes = sum(1 for r in recent_rewards if r > 0.5)
    avg = float(np.mean(recent_rewards))
    rate = successes / len(recent_rewards) * 100.0
    print(f"Success Rate: {rate:.1f}% | Avg Reward: {avg:.3f} | (last {len(recent_rewards)})")

def debug_environment():
    env = GridWorldEnv()
    print("=== Environment Debug ===")
    print(f"Grid: {env.rows} x {env.cols}")
    print(f"Start: {env.start_pos}  Goal: {env.goal_pos}  Trap: {env.trap_pos}  Wall: {env.wall_pos}")
    s, _ = env.reset()
    for a, name in enumerate(['UP', 'RIGHT', 'DOWN', 'LEFT']):
        env.reset()
        ns, r, terminated, truncated, _ = env.step(a)
        print(f" From start action {name:<5} -> {tuple(ns)}, r={r:+.2f}, term={terminated}")



if __name__ == "__main__":
    debug_environment()

    env = GridWorldEnv(step_reward=-0.04)  

    agent = QLearningAgent(
        n_actions=env.action_space.n,
        alpha=0.5,      
        gamma=0.99,
        eps_start=0.3,
        eps_end=0.01,
        eps_decay=0.9995,
    )

    num_episodes = 30000      
    max_steps = 200               
    recent_rewards = []

    # loop for training
    for ep in range(num_episodes):
        state_np, _ = env.reset()
        state = (int(state_np[0]), int(state_np[1]))
        ep_reward = 0.0

        for t in range(max_steps):
            a = agent.choose_action(state)
            next_state_np, reward, terminated, truncated, _ = env.step(a)
            next_state = (int(next_state_np[0]), int(next_state_np[1]))

            agent.update(state, a, reward, next_state, terminated)

            ep_reward += reward
            state = next_state
            if terminated or t == max_steps - 1:
                break

        recent_rewards.append(ep_reward)
        if len(recent_rewards) > 200:
            recent_rewards.pop(0)
        agent.decay_epsilon()

        if (ep + 1) % 5000 == 0:
            print(f"Episode {ep + 1}/{num_episodes} | ε={agent.eps:.3f} | ", end="")
            print_success_stats(recent_rewards)

    print("\n--- Training Finished ---")
    print_success_stats(recent_rewards)

    print_policy(agent.Q, env)
    print_q_table(agent.Q, env)

    print("\n--- Testing Final Policy (20 episodes) ---")
    agent.eps = 0.0
    successes = 0
    for i in range(20):
        state_np, _ = env.reset()
        state = (int(state_np[0]), int(state_np[1]))
        total = 0.0
        for t in range(100):
            # greedy action
            qvals = [agent.Q[(state, a)] for a in range(env.action_space.n)]
            best = np.flatnonzero(qvals == np.max(qvals))
            a = int(np.random.choice(best))
            state_np, r, term, trunc, _ = env.step(a)
            state = (int(state_np[0]), int(state_np[1]))
            total += r
            if term:
                break

        s = state
        outcome = ("SUCCESS" if s == tuple(env.goal_pos)
                   else "TRAP" if s == tuple(env.trap_pos)
                   else "TIMEOUT")
        successes += (outcome == "SUCCESS")
        print(f"Test {i+1:02d}: {outcome} | Reward {total:+.3f}")

    print(f"\nFinal Test Results: {successes}/20 successes ({successes/20*100:.1f}%)")
    env.close()



with open("q_table_eplison_slow.json","w") as f:
    json.dump([{ "state":[int(s[0]),int(s[1])], "action":int(a), "q":float(v)}
               for (s,a),v in agent.Q.items()], f, indent=2)


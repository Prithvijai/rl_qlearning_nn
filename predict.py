import argparse, json, os, sys
import numpy as np
from collections import defaultdict

from env import GridWorldEnv

ARROWS = {0: '↑', 1: '→', 2: '↓', 3: '←'}

def load_qtable(path, n_actions):
    """
    Load Q as a defaultdict(float) keyed by ((r,c), a).
    Accepts several JSON formats:
      1) [{"state":[r,c],"action":a,"q":val}, ...]   (recommended)
      2) {"(r,c),a": val, ...}                       (string keys)
      3) {"(r,c)":{"0":q_up,"1":q_right,"2":q_down,"3":q_left}, ...}
    """
    Q = defaultdict(float)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Q-table file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        # format 1
        for item in data:
            s = tuple(item["state"])
            a = int(item["action"])
            q = float(item["q"])
            Q[(s, a)] = q
        return Q

    if isinstance(data, dict):
        # detect format 3: nested dict of actions
        nested_detect = False
        for k, v in data.items():
            if isinstance(v, dict):
                nested_detect = True
                break

        if nested_detect:
            # format 3
            for sk, adict in data.items():
                # state keys may be "(r, c)" or "r,c"
                if isinstance(sk, str):
                    if sk.startswith("(") and sk.endswith(")"):
                        # "(r, c)" -> (r,c)
                        nums = sk.strip("()").split(",")
                        r, c = int(nums[0]), int(nums[1])
                    else:
                        # "r,c" -> (r,c)
                        nums = sk.split(",")
                        r, c = int(nums[0]), int(nums[1])
                else:
                    # already a tuple/list
                    r, c = sk
                s = (r, c)
                for ak, qv in adict.items():
                    a = int(ak)
                    Q[(s, a)] = float(qv)
            return Q

        # otherwise format 2: flat dict with string keys
        for k, v in data.items():
            # keys like "(r,c),a" or "r,c,a"
            if isinstance(k, str):
                if k.startswith("("):
                    # "(r, c),a"
                    st, a_str = k.split("),")
                    st = st.strip()[1:]  # drop leading "("
                    r, c = [int(x) for x in st.split(",")]
                    a = int(a_str)
                else:
                    # "r,c,a"
                    parts = k.split(",")
                    r, c, a = int(parts[0]), int(parts[1]), int(parts[2])
            else:
                # unexpected, try tuple ((r,c),a)
                (r, c), a = k
            Q[((r, c), int(a))] = float(v)
        return Q

    raise ValueError("Unsupported Q-table JSON structure.")

def greedy_action(Q, state, n_actions):
    qvals = [Q[(state, a)] for a in range(n_actions)]
    best = np.flatnonzero(qvals == np.max(qvals))
    return int(np.random.choice(best)), qvals

def print_policy(Q, env):
    print("\n--- Policy (greedy from Q) ---")
    for r in range(env.rows):
        row = []
        for c in range(env.cols):
            s = (r, c)
            if s == tuple(env.goal_pos):
                row.append("G")
            elif s == tuple(env.trap_pos):
                row.append("T")
            elif s == tuple(env.wall_pos):
                row.append("W")
            else:
                qvals = [Q[(s, a)] for a in range(env.action_space.n)]
                if any(abs(v) > 1e-9 for v in qvals):
                    a = int(np.argmax(qvals))
                    row.append(ARROWS[a])
                else:
                    row.append(".")
        print(" ".join(row))

def run_episode(env, Q, render_mode=None, max_steps=100, verbose=True):
    if render_mode is not None:
        env.render_mode = render_mode

    s_np, _ = env.reset()
    state = (int(s_np[0]), int(s_np[1]))
    total = 0.0

    if verbose:
        print(f"Start at {state}")

    for t in range(max_steps):
        a, qvals = greedy_action(Q, state, env.action_space.n)
        s_np, r, term, trunc, _ = env.step(a)
        ns = (int(s_np[0]), int(s_np[1]))
        total += r

        if verbose:
            print(f"  t={t:02d}  s={state}  a={a}({ARROWS[a]})  r={r:+.2f}  s'={ns}  term={term}")

        state = ns
        if term:
            break

    final = state
    outcome = ("SUCCESS" if final == tuple(env.goal_pos)
               else "TRAP" if final == tuple(env.trap_pos)
               else "TIMEOUT")

    if verbose:
        print(f"Outcome: {outcome} | Return: {total:+.3f}")

    return outcome, total

def main():
    parser = argparse.ArgumentParser(description="Run greedy policy from a saved Q-table on GridWorldEnv.")
    parser.add_argument("--qfile", type=str, default="q_table.json",
                        help="Path to Q-table JSON (default: q_table.json).")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes.")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Max steps per episode.")
    parser.add_argument("--render", type=str, default=None, choices=[None, "human", "ansi"],
                        help="Render mode (pygame 'human' or 'ansi').")
    args = parser.parse_args()

    # Load env and Q
    env = GridWorldEnv(step_reward=-0.04)  # your screenshot spec
    try:
        Q = load_qtable(args.qfile, env.action_space.n)
    except Exception as e:
        print(f"Failed to load Q-table: {e}")
        sys.exit(1)

    print_policy(Q, env)

    successes = 0
    for ep in range(args.episodes):
        print(f"\n=== Eval Episode {ep+1}/{args.episodes} ===")
        outcome, ret = run_episode(env, Q, render_mode=args.render, max_steps=args.max_steps, verbose=True)
        if outcome == "SUCCESS":
            successes += 1

    print(f"\nSuccesses: {successes}/{args.episodes}  ({successes/args.episodes*100:.1f}%)")
    env.close()

if __name__ == "__main__":
    main()

"""Show how CartPoleJs-v1 responds to actions from a state you choose.

Run with:
    uv run python python/probe_env.py --state 0 0 0.05 0 --actions 1 1 1 0 0
    uv run python python/probe_env.py --state 0 0 0.05 0 --actions 1 --repeat 20

State is [x, x_dot, theta, theta_dot]; action 0 pushes left, 1 pushes right.
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np

import cartpole_js  # noqa: F401  registers "CartPoleJs-v1"

ENV_ID = "CartPoleJs-v1"
ACTION_NAMES = {0: "left", 1: "right"}
HEADER = f"{'step':>4}  {'action':>6}  {'x':>8}  {'x_dot':>8}  {'theta':>8}  {'theta_dot':>9}  {'theta°':>7}  status"


def format_row(step: int, action: str, observation: np.ndarray, status: str) -> str:
    x, x_dot, theta, theta_dot = observation
    return (
        f"{step:>4}  {action:>6}  {x:>8.4f}  {x_dot:>8.4f}  {theta:>8.4f}  {theta_dot:>9.4f}"
        f"  {np.degrees(theta):>7.2f}  {status}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", type=float, nargs=4, default=[0.0, 0.0, 0.0, 0.0],
                        metavar=("X", "X_DOT", "THETA", "THETA_DOT"))
    parser.add_argument("--actions", type=int, nargs="+", default=[1], choices=[0, 1])
    parser.add_argument("--repeat", type=int, default=1, help="repeat the action sequence this many times")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = gym.make(ENV_ID)
    try:
        observation, _ = env.reset(options={"state": args.state})
        print(HEADER)
        print(format_row(0, "-", observation, "start"))
        for step, action in enumerate(args.actions * args.repeat, start=1):
            observation, _, terminated, truncated, _ = env.step(action)
            status = "terminated" if terminated else "truncated" if truncated else ""
            print(format_row(step, ACTION_NAMES[action], observation, status))
            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()

"""Train an agent to play CartPoleJs-v1.

Run with: uv run python python/train.py --episodes 500

The env loop, logging, and evaluation are done. Fill in the TODOs in `Agent`
to plug in your learning algorithm (e.g. REINFORCE, DQN, PPO).
Until then the agent acts randomly, so the script runs end to end.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import cartpole_js
import gymnasium as gym
import numpy as np

ENV_ID = "CartPoleJs-v1"
SOLVED_RETURN = 475.0
LOG_EVERY = 10
AVERAGE_WINDOW = 100


@dataclass
class Transition:
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    is_done: bool


class Agent:
    """Policy and learning logic. Observation: [x, x_dot, theta, theta_dot]; action: 0=left, 1=right."""

    def __init__(self, observation_size: int, action_count: int, seed: int) -> None:
        self.observation_size = observation_size
        self.action_count = action_count
        self.rng = np.random.default_rng(seed)
        # TODO: create your model / parameters / optimizer here.

    def act(self, observation: np.ndarray, is_training: bool = True) -> int:
        # TODO: replace with your policy (use is_training to toggle exploration).
        return int(self.rng.integers(self.action_count))

    def observe(self, transition: Transition) -> None:
        # TODO: store the transition (replay buffer, episode trajectory, ...).
        pass

    def end_episode(self) -> None:
        # TODO: run the learning update (per episode for policy gradients, or per step in observe for DQN).
        pass

    def save(self, path: str) -> None:
        # TODO: persist model weights.
        pass


def run_episode(env: gym.Env, agent: Agent, seed: int | None, is_training: bool) -> float:
    observation, _ = env.reset(seed=seed)
    episode_return = 0.0
    is_done = False
    while not is_done:
        action = agent.act(observation, is_training)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        is_done = terminated or truncated
        if is_training:
            # Only `terminated` marks a true end state; truncation should still bootstrap.
            agent.observe(Transition(observation, action, float(reward), next_observation, terminated))
        observation = next_observation
        episode_return += float(reward)
    if is_training:
        agent.end_episode()
    return episode_return


def train(env: gym.Env, agent: Agent, episode_count: int, seed: int) -> list[float]:
    returns: list[float] = []
    for episode in range(1, episode_count + 1):
        # Seed only the first reset; later resets continue the env's RNG stream.
        returns.append(run_episode(env, agent, seed if episode == 1 else None, is_training=True))
        average = float(np.mean(returns[-AVERAGE_WINDOW:]))
        if episode % LOG_EVERY == 0:
            print(f"episode {episode:5d}  return {returns[-1]:6.1f}  avg{AVERAGE_WINDOW} {average:6.1f}")
        if len(returns) >= AVERAGE_WINDOW and average >= SOLVED_RETURN:
            print(f"Solved at episode {episode} (avg{AVERAGE_WINDOW} = {average:.1f})")
            break
    return returns


def evaluate(env: gym.Env, agent: Agent, episode_count: int) -> float:
    returns = [run_episode(env, agent, seed=None, is_training=False) for _ in range(episode_count)]
    return float(np.mean(returns))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", default="cartpole_agent.npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = gym.make(ENV_ID)
    try:
        agent = Agent(env.observation_space.shape[0], env.action_space.n, args.seed)
        train(env, agent, args.episodes, args.seed)
        print(f"Eval avg return over {args.eval_episodes} episodes: {evaluate(env, agent, args.eval_episodes):.1f}")
        agent.save(args.save_path)
    finally:
        env.close()


if __name__ == "__main__":
    main()

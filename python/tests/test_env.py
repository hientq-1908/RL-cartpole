import gymnasium as gym
import numpy as np
import pytest

import cartpole_js  # noqa: F401  registers CartPoleJs-v1
from cartpole_js import CartPoleJsEnv, JsBridgeError

PARITY_STEPS = 50


@pytest.fixture
def env():
    environment = CartPoleJsEnv()
    yield environment
    environment.close()


def test_should_match_gymnasium_physics(env):
    reference = gym.make("CartPole-v1").unwrapped
    observation, _ = env.reset(seed=0)
    reference.reset(seed=0)
    reference.state = observation.astype(np.float64)

    rng = np.random.default_rng(0)
    for _ in range(PARITY_STEPS):
        action = int(rng.integers(2))
        js_obs, _, js_terminated, _, _ = env.step(action)
        ref_obs, _, ref_terminated, _, _ = reference.step(action)
        np.testing.assert_allclose(js_obs, ref_obs, rtol=1e-5, atol=1e-6)
        assert js_terminated == ref_terminated
        if js_terminated:
            break


def test_should_be_reproducible_with_seed(env):
    first, _ = env.reset(seed=123)
    second, _ = env.reset(seed=123)
    np.testing.assert_array_equal(first, second)


def test_should_surface_js_errors(env):
    env.reset(seed=0)
    with pytest.raises(JsBridgeError, match="Invalid action"):
        env._call({"cmd": "step", "action": 5})


def test_should_be_creatable_via_gym_make():
    environment = gym.make("CartPoleJs-v1")
    observation, _ = environment.reset(seed=0)
    assert environment.observation_space.contains(observation)
    environment.close()


def test_should_start_from_given_state(env):
    state = [0.1, -0.2, 0.05, 0.3]
    observation, _ = env.reset(options={"state": state})
    np.testing.assert_allclose(observation, state, rtol=1e-6)


def test_should_step_from_given_state_like_gymnasium(env):
    state = np.array([0.1, -0.2, 0.05, 0.3])
    reference = gym.make("CartPole-v1").unwrapped
    reference.reset(seed=0)
    reference.state = state.copy()
    env.reset(options={"state": state})
    js_obs, *_ = env.step(1)
    ref_obs, *_ = reference.step(1)
    np.testing.assert_allclose(js_obs, ref_obs, rtol=1e-5, atol=1e-6)


def test_should_reject_start_state_outside_bounds(env):
    with pytest.raises(JsBridgeError, match="outside"):
        env.reset(options={"state": [0.0, 0.0, 0.5, 0.0]})


def test_should_init_randomly_without_state(env):
    observation, _ = env.reset(seed=0)
    assert np.all(np.abs(observation) <= 0.05)

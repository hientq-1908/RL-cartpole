from gymnasium.envs.registration import register

from cartpole_js.env import CartPoleJsEnv, JsBridgeError

register(id="CartPoleJs-v1", entry_point="cartpole_js.env:CartPoleJsEnv")

__all__ = ["CartPoleJsEnv", "JsBridgeError"]

"""Gymnasium wrapper that drives the JavaScript CartPole core through a Node subprocess."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

DEFAULT_SERVER_SCRIPT = Path(__file__).resolve().parents[2] / "js" / "server.js"
DEFAULT_MAX_STEPS = 500
OBSERVATION_BOUND = np.finfo(np.float32).max


class JsBridgeError(RuntimeError):
    """Raised when the Node process fails or returns an error response."""


class CartPoleJsEnv(gym.Env):
    """CartPole-v1 compatible env whose physics run in js/cartpole.js."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_steps: int = DEFAULT_MAX_STEPS,
        server_script: Path = DEFAULT_SERVER_SCRIPT,
        node_binary: str = "node",
    ) -> None:
        super().__init__()
        node_path = shutil.which(node_binary)
        if node_path is None:
            raise JsBridgeError(f"Node.js binary '{node_binary}' not found on PATH")
        if not Path(server_script).is_file():
            raise JsBridgeError(f"Server script not found: {server_script}")

        self._process = subprocess.Popen(
            [node_path, str(server_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._call({"cmd": "make", "maxSteps": max_steps})
        spec = self._call({"cmd": "spec"})["spec"]

        # Limits are 2x the termination thresholds, as in Gymnasium's CartPole.
        high = np.array(
            [
                spec["xThreshold"] * 2,
                OBSERVATION_BOUND,
                spec["thetaThreshold"] * 2,
                OBSERVATION_BOUND,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Discrete(spec["actionCount"])

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Pass options={"state": [x, x_dot, theta, theta_dot]} to start from a chosen state."""
        super().reset(seed=seed)
        js_seed = seed if seed is not None else int(self.np_random.integers(2**32))
        request: dict[str, Any] = {"cmd": "reset", "seed": js_seed}
        if options and options.get("state") is not None:
            request["state"] = [float(value) for value in options["state"]]
        response = self._call(request)
        return np.asarray(response["observation"], dtype=np.float32), response["info"]

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        response = self._call({"cmd": "step", "action": int(action)})
        return (
            np.asarray(response["observation"], dtype=np.float32),
            float(response["reward"]),
            bool(response["terminated"]),
            bool(response["truncated"]),
            response["info"],
        )

    def close(self) -> None:
        if self._process.poll() is None:
            self._process.stdin.close()
            self._process.wait(timeout=5)
        super().close()

    def _call(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._process.poll() is not None:
            raise JsBridgeError(
                f"Node process exited with code {self._process.returncode}"
            )
        self._process.stdin.write(json.dumps(request) + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            raise JsBridgeError(f"No response from Node for request {request['cmd']}")
        response = json.loads(line)
        if not response.pop("ok"):
            raise JsBridgeError(f"{request['cmd']} failed: {response['error']}")
        return response

# CartPole (JS)

A CartPole game written in plain JavaScript. It runs in a browser for people to play, and Python can call it through a Gymnasium-compatible interface. The physics match Gymnasium's `CartPole-v1`.

## Play in the browser

```bash
npm run serve           # or: python3 -m http.server 8000
# open http://localhost:8000/web/
```

Hold ← / → to push the cart. The episode ends when the pole tilts past 12°, when the cart leaves the track (±2.4), or after 500 steps.

## Embed in your own page

`js/cartpole.js` has no DOM code. `js/renderer.js` draws a state onto a `<canvas>`.

```html
<canvas id="game" width="600" height="400"></canvas>
<script type="module">
  import { CartPoleEnv } from "./js/cartpole.js";
  import { CartPoleRenderer } from "./js/renderer.js";

  const env = new CartPoleEnv();
  const renderer = new CartPoleRenderer(document.getElementById("game"));
  let { observation } = env.reset();

  setInterval(() => {
    const result = env.step(Math.random() < 0.5 ? 0 : 1);
    observation = result.observation;
    if (result.terminated || result.truncated) observation = env.reset().observation;
    renderer.draw(observation);
  }, 20);
</script>
```

| API | Description |
|-----|-------------|
| `new CartPoleEnv({ maxSteps = 500, seed })` | Create an environment |
| `reset({ seed })` → `{ observation, info }` | Start a new episode |
| `step(action)` → `{ observation, reward, terminated, truncated, info }` | `action`: `0` = left, `1` = right |
| `observation` | `[cartPosition, cartVelocity, poleAngle, poleAngularVelocity]` |

## Use from Python

Requires Node.js 18+ on `PATH`.

```bash
uv sync
```

```python
import gymnasium as gym
import cartpole_js  # registers "CartPoleJs-v1"

env = gym.make("CartPoleJs-v1")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(1)
env.close()
```

Each env starts a `node js/server.js` subprocess and exchanges JSON lines over stdin/stdout:

```
→ {"cmd": "reset", "seed": 0}
← {"ok": true, "observation": [...], "info": {}}
→ {"cmd": "step", "action": 1}
← {"ok": true, "observation": [...], "reward": 1, "terminated": false, "truncated": false, "info": {"step": 1}}
```

Any language that can start a process and read and write lines can use the same protocol.

## Tests

```bash
npm test          # JS unit tests
uv run pytest     # Python bridge, including step-by-step parity with Gymnasium CartPole-v1
```

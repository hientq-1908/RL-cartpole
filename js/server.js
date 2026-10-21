// JSON-lines bridge so Python (or any language) can drive the JS environment.
// Each stdin line is a request: {"cmd": "reset"|"step"|"spec"|"close", ...}
// Each stdout line is a response: {"ok": true, ...} or {"ok": false, "error": "..."}

import { createInterface } from "node:readline";

import { CartPoleEnv, SPEC } from "./cartpole.js";

let env = new CartPoleEnv();

function handle(request) {
  switch (request.cmd) {
    case "make":
      env = new CartPoleEnv({ maxSteps: request.maxSteps, seed: request.seed });
      return {};
    case "reset":
      return env.reset({ seed: request.seed, state: request.state });
    case "step":
      return env.step(request.action);
    case "spec":
      return { spec: SPEC };
    case "close":
      process.exit(0);
    default:
      throw new Error(`Unknown command: ${request.cmd}`);
  }
}

const input = createInterface({ input: process.stdin });

input.on("line", (line) => {
  let response;
  try {
    response = { ok: true, ...handle(JSON.parse(line)) };
  } catch (err) {
    response = { ok: false, error: err.message };
  }
  process.stdout.write(JSON.stringify(response) + "\n");
});

input.on("close", () => process.exit(0));

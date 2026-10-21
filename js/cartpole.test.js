import assert from "node:assert/strict";
import { test } from "node:test";

import { CartPoleEnv, ACTION_LEFT, ACTION_RIGHT } from "./cartpole.js";

test("reset with the same seed gives the same observation", () => {
  const first = new CartPoleEnv().reset({ seed: 42 }).observation;
  const second = new CartPoleEnv().reset({ seed: 42 }).observation;
  assert.deepEqual(first, second);
});

test("pushing one direction eventually terminates the episode", () => {
  const env = new CartPoleEnv({ seed: 1 });
  env.reset();
  let result;
  do {
    result = env.step(ACTION_RIGHT);
  } while (!result.terminated && !result.truncated);
  assert.equal(result.terminated, true);
});

test("episode truncates at maxSteps", () => {
  const env = new CartPoleEnv({ maxSteps: 3, seed: 1 });
  env.reset();
  env.step(ACTION_LEFT);
  env.step(ACTION_RIGHT);
  const result = env.step(ACTION_LEFT);
  assert.equal(result.truncated, true);
});

test("rejects invalid actions", () => {
  const env = new CartPoleEnv();
  env.reset();
  assert.throws(() => env.step(2), RangeError);
});

test("rejects step after episode end", () => {
  const env = new CartPoleEnv({ maxSteps: 1 });
  env.reset();
  env.step(ACTION_LEFT);
  assert.throws(() => env.step(ACTION_LEFT), /reset/);
});

test("reset starts from a given state", () => {
  const state = [0.1, -0.2, 0.05, 0.3];
  assert.deepEqual(new CartPoleEnv().reset({ state }).observation, state);
});

test("rejects a malformed reset state", () => {
  assert.throws(() => new CartPoleEnv().reset({ state: [0, 0] }), TypeError);
  assert.throws(() => new CartPoleEnv().reset({ state: [0, NaN, 0, 0] }), TypeError);
});

test("rejects a reset state outside the termination bounds", () => {
  assert.throws(() => new CartPoleEnv().reset({ state: [2.5, 0, 0, 0] }), RangeError);
  assert.throws(() => new CartPoleEnv().reset({ state: [0, 0, 0.3, 0] }), RangeError);
});

test("random reset state stays within ±0.05", () => {
  const env = new CartPoleEnv({ seed: 7 });
  for (let i = 0; i < 100; i += 1) {
    assert.ok(env.reset().observation.every((value) => Math.abs(value) <= 0.05));
  }
});

// CartPole environment core. No DOM dependencies, so it runs in the browser and in Node.
// Physics and constants match Gymnasium's CartPole-v1 so trained policies transfer.

const GRAVITY = 9.8;
const CART_MASS = 1.0;
const POLE_MASS = 0.1;
const TOTAL_MASS = CART_MASS + POLE_MASS;
const POLE_HALF_LENGTH = 0.5;
const POLE_MASS_LENGTH = POLE_MASS * POLE_HALF_LENGTH;
const FORCE_MAGNITUDE = 10.0;
const TIME_STEP = 0.02;
const INITIAL_STATE_RANGE = 0.05;

export const X_THRESHOLD = 2.4;
export const THETA_THRESHOLD = (12 * 2 * Math.PI) / 360;
export const DEFAULT_MAX_STEPS = 500;

export const ACTION_LEFT = 0;
export const ACTION_RIGHT = 1;

// Small seedable PRNG (mulberry32) so episodes are reproducible from Python.
function createRng(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomSeed() {
  return Math.floor(Math.random() * 2 ** 32);
}

// A start state must be finite and inside the termination bounds, otherwise the episode is already over.
function validateState(state) {
  if (!Array.isArray(state) || state.length !== 4 || !state.every(Number.isFinite)) {
    throw new TypeError("state must be an array of 4 finite numbers [x, xDot, theta, thetaDot]");
  }
  const [x, , theta] = state;
  if (Math.abs(x) > X_THRESHOLD) {
    throw new RangeError(`state x=${x} is outside ±${X_THRESHOLD}`);
  }
  if (Math.abs(theta) > THETA_THRESHOLD) {
    throw new RangeError(`state theta=${theta} is outside ±${THETA_THRESHOLD.toFixed(4)} rad (12°)`);
  }
}

export class CartPoleEnv {
  constructor({ maxSteps = DEFAULT_MAX_STEPS, seed } = {}) {
    this.maxSteps = maxSteps;
    this.rng = createRng(seed ?? randomSeed());
    this.state = [0, 0, 0, 0];
    this.stepCount = 0;
    this.isDone = true;
  }

  /**
   * Starts a new episode. Returns { observation, info }.
   * Optional `state` ([x, xDot, theta, thetaDot]) starts from that exact state; it must be inside the
   * termination bounds. Without it, each value is drawn uniformly from ±0.05, as in Gymnasium.
   */
  reset({ seed, state } = {}) {
    if (seed !== undefined && seed !== null) {
      this.rng = createRng(seed);
    }
    if (state !== undefined && state !== null) {
      validateState(state);
      this.state = [...state];
    } else {
      this.state = this.state.map(
        () => (this.rng() * 2 - 1) * INITIAL_STATE_RANGE,
      );
    }
    this.stepCount = 0;
    this.isDone = false;
    return { observation: [...this.state], info: {} };
  }

  /**
   * Advances the simulation by one tick.
   * @param {number} action 0 pushes the cart left, 1 pushes it right.
   * @returns {{observation: number[], reward: number, terminated: boolean, truncated: boolean, info: object}}
   */
  step(action) {
    if (action !== ACTION_LEFT && action !== ACTION_RIGHT) {
      throw new RangeError(`Invalid action ${action}; expected 0 or 1`);
    }
    if (this.isDone) {
      throw new Error("Episode is over; call reset() before step()");
    }

    this.state = this.#integrate(action);
    this.stepCount += 1;

    const [x, , theta] = this.state;
    const terminated =
      Math.abs(x) > X_THRESHOLD || Math.abs(theta) > THETA_THRESHOLD;
    const truncated = !terminated && this.stepCount >= this.maxSteps;
    this.isDone = terminated || truncated;

    return {
      observation: [...this.state],
      reward: 1.0,
      terminated,
      truncated,
      info: { step: this.stepCount },
    };
  }

  // Explicit Euler integration, same as Gymnasium's default.
  #integrate(action) {
    const [x, xDot, theta, thetaDot] = this.state;
    const force = action === ACTION_RIGHT ? FORCE_MAGNITUDE : -FORCE_MAGNITUDE;
    const cosTheta = Math.cos(theta);
    const sinTheta = Math.sin(theta);

    const temp =
      (force + POLE_MASS_LENGTH * thetaDot ** 2 * sinTheta) / TOTAL_MASS;
    const thetaAcc =
      (GRAVITY * sinTheta - cosTheta * temp) /
      (POLE_HALF_LENGTH *
        (4.0 / 3.0 - (POLE_MASS * cosTheta ** 2) / TOTAL_MASS));
    const xAcc = temp - (POLE_MASS_LENGTH * thetaAcc * cosTheta) / TOTAL_MASS;

    return [
      x + TIME_STEP * xDot,
      xDot + TIME_STEP * xAcc,
      theta + TIME_STEP * thetaDot,
      thetaDot + TIME_STEP * thetaAcc,
    ];
  }
}

export const SPEC = {
  observationSize: 4,
  actionCount: 2,
  timeStep: TIME_STEP,
  xThreshold: X_THRESHOLD,
  thetaThreshold: THETA_THRESHOLD,
  poleHalfLength: POLE_HALF_LENGTH,
};

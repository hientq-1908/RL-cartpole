// Canvas renderer for CartPoleEnv. Browser-only.

import { SPEC } from "./cartpole.js";

const WORLD_WIDTH = SPEC.xThreshold * 2;
const POLE_LENGTH = SPEC.poleHalfLength * 2;
const CART_WIDTH_RATIO = 0.1;
const CART_HEIGHT_RATIO = 0.06;
const POLE_WIDTH_RATIO = 0.016;
const TRACK_Y_RATIO = 0.72;

const DEFAULT_COLORS = {
  background: "#ffffff",
  track: "#1f2328",
  cart: "#1f2328",
  pole: "#c98a4b",
  axle: "#7f8fbf",
};

export class CartPoleRenderer {
  constructor(canvas, colors = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.colors = { ...DEFAULT_COLORS, ...colors };
  }

  draw(state) {
    const { ctx, canvas, colors } = this;
    const width = canvas.width;
    const height = canvas.height;
    const scale = width / WORLD_WIDTH;
    const [x, , theta] = state;

    const trackY = height * TRACK_Y_RATIO;
    const cartWidth = width * CART_WIDTH_RATIO;
    const cartHeight = width * CART_HEIGHT_RATIO;
    const cartX = width / 2 + x * scale;
    const axleY = trackY - cartHeight * 0.75;

    ctx.fillStyle = colors.background;
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = colors.track;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, trackY);
    ctx.lineTo(width, trackY);
    ctx.stroke();

    ctx.fillStyle = colors.cart;
    ctx.fillRect(
      cartX - cartWidth / 2,
      trackY - cartHeight,
      cartWidth,
      cartHeight,
    );

    ctx.save();
    ctx.translate(cartX, axleY);
    ctx.rotate(theta);
    const poleWidth = width * POLE_WIDTH_RATIO;
    ctx.fillStyle = colors.pole;
    ctx.fillRect(-poleWidth / 2, -POLE_LENGTH * scale, poleWidth, POLE_LENGTH * scale);
    ctx.restore();

    ctx.fillStyle = colors.axle;
    ctx.beginPath();
    ctx.arc(cartX, axleY, poleWidth / 2, 0, Math.PI * 2);
    ctx.fill();
  }
}

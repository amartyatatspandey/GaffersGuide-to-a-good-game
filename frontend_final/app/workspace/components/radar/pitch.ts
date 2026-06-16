import {
  PITCH_LENGTH_M,
  PITCH_WIDTH_M,
  projectMetersToCanvas,
} from "@/lib/trackingAdapter";

const PITCH_GREEN_TOP = "#0d2b15";
const PITCH_GREEN_BOTTOM = "#061a0d";
const PITCH_LINE = "rgba(255, 255, 255, 0.8)";
const PITCH_LINE_GLOW = "rgba(16, 185, 129, 0.2)";
const LINE_WIDTH = 1.5;

function strokeRectMeters(
  ctx: CanvasRenderingContext2D,
  xM: number,
  yM: number,
  wM: number,
  hM: number,
  widthPx: number,
  heightPx: number,
  paddingPx: number,
): void {
  const topLeft = projectMetersToCanvas(xM, yM, widthPx, heightPx, paddingPx);
  const bottomRight = projectMetersToCanvas(
    xM + wM,
    yM + hM,
    widthPx,
    heightPx,
    paddingPx,
  );
  ctx.strokeRect(
    topLeft.xPx,
    topLeft.yPx,
    bottomRight.xPx - topLeft.xPx,
    bottomRight.yPx - topLeft.yPx,
  );
}

export function drawPitchBackground(
  ctx: CanvasRenderingContext2D,
  widthPx: number,
  heightPx: number,
  paddingPx: number,
): void {
  // 1. Clear and create premium gradient background
  ctx.clearRect(0, 0, widthPx, heightPx);
  const gradient = ctx.createRadialGradient(
    widthPx / 2, heightPx / 2, 0,
    widthPx / 2, heightPx / 2, widthPx / 1.5
  );
  gradient.addColorStop(0, PITCH_GREEN_TOP);
  gradient.addColorStop(1, PITCH_GREEN_BOTTOM);
  
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, widthPx, heightPx);

  // 2. Add subtle grid pattern
  ctx.strokeStyle = "rgba(255, 255, 255, 0.03)";
  ctx.lineWidth = 0.5;
  for (let i = 0; i < 10; i++) {
    const x = paddingPx + (i * (widthPx - paddingPx * 2)) / 9;
    ctx.beginPath();
    ctx.moveTo(x, paddingPx);
    ctx.lineTo(x, heightPx - paddingPx);
    ctx.stroke();
  }

  // 3. Set line styles
  ctx.strokeStyle = PITCH_LINE;
  ctx.lineWidth = LINE_WIDTH;
  ctx.shadowBlur = 4;
  ctx.shadowColor = PITCH_LINE_GLOW;

  // Outer boundary.
  strokeRectMeters(
    ctx,
    0,
    0,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    widthPx,
    heightPx,
    paddingPx,
  );

  // Halfway line.
  const centerTop = projectMetersToCanvas(
    PITCH_LENGTH_M / 2,
    0,
    widthPx,
    heightPx,
    paddingPx,
  );
  const centerBottom = projectMetersToCanvas(
    PITCH_LENGTH_M / 2,
    PITCH_WIDTH_M,
    widthPx,
    heightPx,
    paddingPx,
  );
  ctx.beginPath();
  ctx.moveTo(centerTop.xPx, centerTop.yPx);
  ctx.lineTo(centerBottom.xPx, centerBottom.yPx);
  ctx.stroke();

  // Center circle and spot.
  const center = projectMetersToCanvas(
    PITCH_LENGTH_M / 2,
    PITCH_WIDTH_M / 2,
    widthPx,
    heightPx,
    paddingPx,
  );
  const edgePoint = projectMetersToCanvas(
    PITCH_LENGTH_M / 2 + 9.15,
    PITCH_WIDTH_M / 2,
    widthPx,
    heightPx,
    paddingPx,
  );
  const circleR = Math.max(1, edgePoint.xPx - center.xPx);
  
  ctx.beginPath();
  ctx.arc(center.xPx, center.yPx, circleR, 0, Math.PI * 2);
  ctx.stroke();
  
  ctx.beginPath();
  ctx.fillStyle = PITCH_LINE;
  ctx.arc(center.xPx, center.yPx, 1.5, 0, Math.PI * 2);
  ctx.fill();

  // Penalty areas.
  const penAreaDepth = 16.5;
  const penAreaWidth = 40.3;
  const penY = (PITCH_WIDTH_M - penAreaWidth) / 2;
  
  strokeRectMeters(ctx, 0, penY, penAreaDepth, penAreaWidth, widthPx, heightPx, paddingPx);
  strokeRectMeters(ctx, PITCH_LENGTH_M - penAreaDepth, penY, penAreaDepth, penAreaWidth, widthPx, heightPx, paddingPx);

  // Six-yard boxes.
  const sixDepth = 5.5;
  const sixWidth = 18.32;
  const sixY = (PITCH_WIDTH_M - sixWidth) / 2;
  
  strokeRectMeters(ctx, 0, sixY, sixDepth, sixWidth, widthPx, heightPx, paddingPx);
  strokeRectMeters(ctx, PITCH_LENGTH_M - sixDepth, sixY, sixDepth, sixWidth, widthPx, heightPx, paddingPx);

  // Reset shadow for players
  ctx.shadowBlur = 0;
}

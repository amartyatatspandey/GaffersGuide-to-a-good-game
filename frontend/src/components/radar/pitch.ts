import {
  PITCH_LENGTH_M,
  PITCH_WIDTH_M,
  projectMetersToCanvas,
} from "../../lib/trackingAdapter";

const PITCH_GREEN = "#1f7a33";
const PITCH_LINE = "#ffffff";
const LINE_WIDTH = 2;

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
  ctx.clearRect(0, 0, widthPx, heightPx);
  ctx.fillStyle = PITCH_GREEN;
  ctx.fillRect(0, 0, widthPx, heightPx);

  ctx.strokeStyle = PITCH_LINE;
  ctx.lineWidth = LINE_WIDTH;

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
  ctx.arc(center.xPx, center.yPx, 2, 0, Math.PI * 2);
  ctx.fill();

  // Penalty areas.
  const penAreaDepth = 16.5;
  const penAreaWidth = 40.3;
  const penY = (PITCH_WIDTH_M - penAreaWidth) / 2;
  strokeRectMeters(
    ctx,
    0,
    penY,
    penAreaDepth,
    penAreaWidth,
    widthPx,
    heightPx,
    paddingPx,
  );
  strokeRectMeters(
    ctx,
    PITCH_LENGTH_M - penAreaDepth,
    penY,
    penAreaDepth,
    penAreaWidth,
    widthPx,
    heightPx,
    paddingPx,
  );

  // Six-yard boxes.
  const sixDepth = 5.5;
  const sixWidth = 18.32;
  const sixY = (PITCH_WIDTH_M - sixWidth) / 2;
  strokeRectMeters(
    ctx,
    0,
    sixY,
    sixDepth,
    sixWidth,
    widthPx,
    heightPx,
    paddingPx,
  );
  strokeRectMeters(
    ctx,
    PITCH_LENGTH_M - sixDepth,
    sixY,
    sixDepth,
    sixWidth,
    widthPx,
    heightPx,
    paddingPx,
  );
}


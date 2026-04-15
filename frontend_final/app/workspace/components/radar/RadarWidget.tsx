import { RefObject, useEffect, useMemo, useState } from "react";
import { useVideoFrameSync } from "../../hooks/useVideoFrameSync";
import { buildFrameLookup } from "../../lib/trackingAdapter";
import type { TrackingFrame, TrackingPayload } from "../../lib/trackingTypes";
import RadarCanvas from "./RadarCanvas";

interface RadarWidgetProps {
  videoRef: RefObject<HTMLVideoElement | null>;
  trackingData: TrackingPayload | null;
  onFrameChange?: (frameIndex: number, frame: TrackingFrame | undefined) => void;
}

export default function RadarWidget({
  videoRef,
  trackingData,
  onFrameChange,
}: RadarWidgetProps) {
  const lookup = useMemo(
    () => (trackingData ? buildFrameLookup(trackingData) : null),
    [trackingData],
  );
  const [currentFrame, setCurrentFrame] = useState(0);
  const [syncFps, setSyncFps] = useState(30);

  useEffect(() => {
    setCurrentFrame(0);
  }, [trackingData]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !trackingData) return;
    const totalFrames =
      trackingData.telemetry?.total_frames_processed ?? trackingData.frames.length;
    const onLoadedMetadata = (): void => {
      if (video.duration > 0 && totalFrames > 0) {
        const derived = totalFrames / video.duration;
        setSyncFps(Number.isFinite(derived) && derived > 0 ? derived : 30);
      } else {
        setSyncFps(30);
      }
    };
    video.addEventListener("loadedmetadata", onLoadedMetadata);
    onLoadedMetadata();
    return () => video.removeEventListener("loadedmetadata", onLoadedMetadata);
  }, [videoRef, trackingData]);

  useVideoFrameSync({
    videoRef,
    fps: syncFps,
    maxFrameIndex: Math.max(0, (lookup?.totalFrames ?? 1) - 1),
    onFrameChange: setCurrentFrame,
  });

  const frame = lookup?.getFrameByIndex(currentFrame);

  useEffect(() => {
    onFrameChange?.(currentFrame, frame);
  }, [currentFrame, frame, onFrameChange]);

  return (
    <div className="rounded-lg border bg-white p-3 shadow-sm">
      <h2 className="mb-2 text-sm font-semibold text-gray-800">Tactical Radar</h2>
      <RadarCanvas frame={frame} />
      <div className="mt-2 flex items-center justify-between text-xs text-gray-500">
        <span>Frame {currentFrame}</span>
        <span>Sync FPS {syncFps.toFixed(2)}</span>
      </div>
    </div>
  );
}


import type { ChangeEvent } from "react";
import { RefObject, useCallback, useEffect, useState, useMemo, useRef } from "react";

function formatClock(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) {
    return `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
  }
  return `${m}:${String(sec).padStart(2, "0")}`;
}

interface VideoHUDProps {
  videoRef: RefObject<HTMLVideoElement | null>;
  videoSrc: string | null;
  jobId: string | null;
  status: string;
  onDownload: () => void;
  trackingData: any;
  useAltNames?: boolean;
  dictionary?: Record<string, string>;
}

export default function VideoHUD({
  videoRef,
  videoSrc,
  jobId,
  status,
  onDownload,
  trackingData,
  useAltNames = false,
  dictionary = {},
}: VideoHUDProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [showOverlays, setShowOverlays] = useState(true);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const [geometry, setGeometry] = useState({
    width: 0,
    height: 0,
    left: 0,
    top: 0,
    scaleX: 1,
    scaleY: 1,
    videoWidth: 1920,
    videoHeight: 1080
  });

  const updateGeometry = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    
    const rect = video.getBoundingClientRect();
    const videoW = video.videoWidth || 1920;
    const videoH = video.videoHeight || 1080;
    
    const containerRatio = rect.width / rect.height;
    const videoRatio = videoW / videoH;
    
    let displayedW = rect.width;
    let displayedH = rect.height;
    let left = 0;
    let top = 0;
    
    if (containerRatio > videoRatio) {
      displayedW = rect.height * videoRatio;
      left = (rect.width - displayedW) / 2;
    } else {
      displayedH = rect.width / videoRatio;
      top = (rect.height - displayedH) / 2;
    }
    
    setGeometry({
      width: displayedW,
      height: displayedH,
      left: left,
      top: top,
      scaleX: displayedW / videoW,
      scaleY: displayedH / videoH,
      videoWidth: videoW,
      videoHeight: videoH
    });
  }, [videoRef]);

  // Keep geometry updated on resize or load
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    
    const observer = new ResizeObserver(() => {
      updateGeometry();
    });
    observer.observe(video);
    
    video.addEventListener("loadedmetadata", updateGeometry);
    
    return () => {
      observer.disconnect();
      video.removeEventListener("loadedmetadata", updateGeometry);
    };
  }, [videoRef, videoSrc, updateGeometry]);

  const toggleFullscreen = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;

    if (!document.fullscreenElement) {
      container.requestFullscreen?.().catch((err) => {
        console.error(`Error attempting to enable fullscreen: ${err.message}`);
      });
    } else {
      document.exitFullscreen?.();
    }
  }, []);

  useEffect(() => {
    const onFullscreenChange = () => {
      const isFull = document.fullscreenElement === containerRef.current;
      setIsFullscreen(isFull);
      updateGeometry();
      setTimeout(updateGeometry, 50);
      setTimeout(updateGeometry, 150);
      setTimeout(updateGeometry, 300);
    };

    document.addEventListener("fullscreenchange", onFullscreenChange);
    document.addEventListener("webkitfullscreenchange", onFullscreenChange);
    document.addEventListener("mozfullscreenchange", onFullscreenChange);
    document.addEventListener("MSFullscreenChange", onFullscreenChange);

    return () => {
      document.removeEventListener("fullscreenchange", onFullscreenChange);
      document.removeEventListener("webkitfullscreenchange", onFullscreenChange);
      document.removeEventListener("mozfullscreenchange", onFullscreenChange);
      document.removeEventListener("MSFullscreenChange", onFullscreenChange);
    };
  }, [updateGeometry]);

  const activeFrame = useMemo(() => {
    if (!trackingData?.frames || trackingData.frames.length === 0) return null;
    const fps = trackingData.telemetry?.fps || 25; // default to 25
    const targetIdx = Math.round(currentTime * fps);
    
    // Binary search for the closest frame index
    let closestFrame = trackingData.frames[0];
    let minDiff = Math.abs(closestFrame.frame_idx - targetIdx);
    
    let low = 0;
    let high = trackingData.frames.length - 1;
    while (low <= high) {
      const mid = Math.floor((low + high) / 2);
      const diff = Math.abs(trackingData.frames[mid].frame_idx - targetIdx);
      if (diff < minDiff) {
        minDiff = diff;
        closestFrame = trackingData.frames[mid];
      }
      if (trackingData.frames[mid].frame_idx < targetIdx) {
        low = mid + 1;
      } else {
        high = mid - 1;
      }
    }
    return closestFrame;
  }, [trackingData, currentTime]);

  const filteredPlayers = useMemo(() => {
    if (!activeFrame?.players) return [];
    return activeFrame.players.filter((p: any) => p.id !== 75 && p.id !== 76 && p.id !== 79);
  }, [activeFrame]);

  const backLineDefenders = useMemo(() => {
    if (filteredPlayers.length === 0) return { team0: [], team1: [] };
    
    const team0 = filteredPlayers.filter((p: any) => p.team_id === 'team_0' || p.team === 'team_0');
    const team1 = filteredPlayers.filter((p: any) => p.team_id === 'team_1' || p.team === 'team_1');
    
    const team0Sorted = [...team0].filter(p => p.x_pitch !== null && p.y_pitch !== null)
      .sort((a, b) => (a.x_pitch ?? 0) - (b.x_pitch ?? 0));
    const team1Sorted = [...team1].filter(p => p.x_pitch !== null && p.y_pitch !== null)
      .sort((a, b) => (b.x_pitch ?? 0) - (a.x_pitch ?? 0));
      
    // Skip goalkeeper (index 0) and take next 4
    const defenders0 = team0Sorted.slice(1, 5);
    const defenders1 = team1Sorted.slice(1, 5);
    
    // Sort by y_pitch to draw a line connecting them from top to bottom
    defenders0.sort((a, b) => (a.y_pitch ?? 0) - (b.y_pitch ?? 0));
    defenders1.sort((a, b) => (a.y_pitch ?? 0) - (b.y_pitch ?? 0));
    
    return { team0: defenders0, team1: defenders1 };
  }, [filteredPlayers]);

  const getPitchDistance = useCallback((p1: [number, number], p2: [number, number]) => {
    let dx = p1[0] - p2[0];
    let dy = p1[1] - p2[1];
    if (Math.abs(p1[0]) > 110 || Math.abs(p2[0]) > 110) {
      dx = dx / 10;
      dy = dy / 10;
    }
    return Math.hypot(dx, dy);
  }, []);

  const ballArrowPath = useMemo(() => {
    if (!trackingData?.frames || !activeFrame) return null;
    const currentIdx = activeFrame.frame_idx;
    
    const activeFrameIndex = trackingData.frames.findIndex((f: any) => f.frame_idx === currentIdx);
    if (activeFrameIndex === -1) return null;

    const startIndex = Math.max(0, activeFrameIndex - 30);
    const pathPoints: { x: number; y: number; frame_idx: number; ball_xy: [number, number] | null }[] = [];

    for (let i = startIndex; i <= activeFrameIndex; i++) {
      const f = trackingData.frames[i];
      if (f.ball_canvas && Array.isArray(f.ball_canvas) && f.ball_canvas.length >= 2) {
        const [bx, by] = f.ball_canvas;
        if (bx !== null && by !== null) {
          pathPoints.push({
            x: bx * geometry.scaleX,
            y: by * geometry.scaleY,
            frame_idx: f.frame_idx,
            ball_xy: f.ball_xy
          });
        }
      }
    }
    
    return pathPoints;
  }, [trackingData, activeFrame, geometry.scaleX, geometry.scaleY]);

  const passArrow = useMemo(() => {
    if (!ballArrowPath || ballArrowPath.length < 5) return null;
    
    const startPt = ballArrowPath[0];
    const endPt = ballArrowPath[ballArrowPath.length - 1];
    
    const canvasDist = Math.hypot(endPt.x - startPt.x, endPt.y - startPt.y);
    if (canvasDist < 40) return null;
    
    const midX = (startPt.x + endPt.x) / 2;
    const midY = (startPt.y + endPt.y) / 2;
    
    const heightOffset = Math.min(120, canvasDist * 0.35);
    const ctrlX = midX;
    const ctrlY = midY - heightOffset;
    
    const peakX = 0.25 * startPt.x + 0.5 * ctrlX + 0.25 * endPt.x;
    const peakY = 0.25 * startPt.y + 0.5 * ctrlY + 0.25 * endPt.y;
    
    const angle = Math.atan2(endPt.y - ctrlY, endPt.x - ctrlX);
    
    const arrowLength = 12;
    const arrowAngle = Math.PI / 6;
    const xLeft = endPt.x - arrowLength * Math.cos(angle - arrowAngle);
    const yLeft = endPt.y - arrowLength * Math.sin(angle - arrowAngle);
    const xRight = endPt.x - arrowLength * Math.cos(angle + arrowAngle);
    const yRight = endPt.y - arrowLength * Math.sin(angle + arrowAngle);
    
    let distanceMeters = 0;
    if (startPt.ball_xy && endPt.ball_xy) {
      distanceMeters = getPitchDistance(startPt.ball_xy, endPt.ball_xy);
    } else {
      distanceMeters = canvasDist * 0.15;
    }
    
    return {
      d: `M ${startPt.x} ${startPt.y} Q ${ctrlX} ${ctrlY} ${endPt.x} ${endPt.y}`,
      arrowPoints: `${endPt.x},${endPt.y} ${xLeft},${yLeft} ${xRight},${yRight}`,
      peak: { x: peakX, y: peakY },
      distance: distanceMeters,
    };
  }, [ballArrowPath, getPitchDistance]);

  const tacticalTriangles = useMemo(() => {
    if (filteredPlayers.length === 0) return [];
    
    const teams = ['team_0', 'team_1'];
    const results: { pointsStr: string; color: string; perimeter: number }[] = [];
    
    teams.forEach(teamId => {
      const teamPlayers = filteredPlayers.filter(
        (p: any) => (p.team_id === teamId || p.team === teamId) && p.x_pitch != null && p.y_pitch != null
      );
      
      const n = teamPlayers.length;
      const validTriplets: { points: {x: number, y: number}[]; perimeter: number }[] = [];
      
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          for (let k = j + 1; k < n; k++) {
            const pA = teamPlayers[i];
            const pB = teamPlayers[j];
            const pC = teamPlayers[k];
            
            const d_AB = getPitchDistance([pA.x_pitch, pA.y_pitch], [pB.x_pitch, pB.y_pitch]);
            const d_BC = getPitchDistance([pB.x_pitch, pB.y_pitch], [pC.x_pitch, pC.y_pitch]);
            const d_CA = getPitchDistance([pC.x_pitch, pC.y_pitch], [pA.x_pitch, pA.y_pitch]);
            
            const minDist = 5;
            const maxDist = 22;
            if (
              d_AB >= minDist && d_AB <= maxDist &&
              d_BC >= minDist && d_BC <= maxDist &&
              d_CA >= minDist && d_CA <= maxDist
            ) {
              const ptA = {
                x: pA.x_canvas * geometry.scaleX,
                y: pA.bbox ? pA.bbox[3] * geometry.scaleY : (pA.y_canvas + 15) * geometry.scaleY
              };
              const ptB = {
                x: pB.x_canvas * geometry.scaleX,
                y: pB.bbox ? pB.bbox[3] * geometry.scaleY : (pB.y_canvas + 15) * geometry.scaleY
              };
              const ptC = {
                x: pC.x_canvas * geometry.scaleX,
                y: pC.bbox ? pC.bbox[3] * geometry.scaleY : (pC.y_canvas + 15) * geometry.scaleY
              };
              
              validTriplets.push({
                points: [ptA, ptB, ptC],
                perimeter: d_AB + d_BC + d_CA
              });
            }
          }
        }
      }
      
      validTriplets.sort((a, b) => a.perimeter - b.perimeter);
      
      const color = teamId === 'team_0' ? '#ef4444' : '#3b82f6';
      const countToTake = Math.min(2, validTriplets.length);
      for (let idx = 0; idx < countToTake; idx++) {
        const triplet = validTriplets[idx];
        const ptsStr = triplet.points.map(p => `${p.x},${p.y}`).join(' ');
        results.push({
          pointsStr: ptsStr,
          color,
          perimeter: triplet.perimeter
        });
      }
    });
    
    return results;
  }, [filteredPlayers, geometry, getPitchDistance]);

  const detectedClips = useMemo(() => {
    if (!trackingData?.frames || trackingData.frames.length === 0) {
      return [
        { id: 'pass-demo', title: 'Passing Arc', desc: '3D Passing Trajectory Arc', time: 9, type: 'pass' },
        { id: 'tri-demo', title: 'Support Triangle', desc: 'Teammate Passing Network', time: 13, type: 'triangle' },
        { id: 'switch-demo', title: 'Flank Switch', desc: 'Long Curved Flank Switch', time: 38, type: 'pass' },
        { id: 'press-demo', title: 'Defensive Press', desc: 'Defensive Line & Free Man', time: 71, type: 'defense' }
      ];
    }
    
    const clips: { id: string; title: string; desc: string; time: number; type: 'pass' | 'triangle' | 'defense' }[] = [];
    const frames = trackingData.frames;
    const fps = trackingData.telemetry?.fps || 25;
    
    const step = Math.max(1, Math.round(fps));
    
    let foundPass = false;
    let foundTriangle = false;
    
    for (let i = 0; i < frames.length; i += step) {
      const f = frames[i];
      const timeSec = f.frame_idx / fps;
      
      const team0Players = f.players.filter((p: any) => p.team_id === 'team_0' && p.x_pitch != null && p.y_pitch != null);
      const team1Players = f.players.filter((p: any) => p.team_id === 'team_1' && p.x_pitch != null && p.y_pitch != null);
      
      const hasTri = (players: any[]) => {
        const n = players.length;
        for (let a = 0; a < n; a++) {
          for (let b = a + 1; b < n; b++) {
            for (let c = b + 1; c < n; c++) {
              const d1 = getPitchDistance([players[a].x_pitch, players[a].y_pitch], [players[b].x_pitch, players[b].y_pitch]);
              const d2 = getPitchDistance([players[b].x_pitch, players[b].y_pitch], [players[c].x_pitch, players[c].y_pitch]);
              const d3 = getPitchDistance([players[c].x_pitch, players[c].y_pitch], [players[a].x_pitch, players[a].y_pitch]);
              if (d1 >= 5 && d1 <= 22 && d2 >= 5 && d2 <= 22 && d3 >= 5 && d3 <= 22) {
                return true;
              }
            }
          }
        }
        return false;
      };
      
      if (!foundTriangle && (hasTri(team0Players) || hasTri(team1Players))) {
        clips.push({
          id: `tri-${f.frame_idx}`,
          title: 'Support Network',
          desc: 'Active teammate support triangle',
          time: Math.round(timeSec),
          type: 'triangle'
        });
        foundTriangle = true;
      }
      
      if (!foundPass && f.ball_canvas) {
        const lookbackIdx = Math.max(0, i - 15);
        const prevF = frames[lookbackIdx];
        if (prevF.ball_canvas) {
          const dist = Math.hypot(f.ball_canvas[0] - prevF.ball_canvas[0], f.ball_canvas[1] - prevF.ball_canvas[1]);
          if (dist > 80) {
            clips.push({
              id: `pass-${f.frame_idx}`,
              title: 'Passing Play',
              desc: 'Ball trajectory arc detected',
              time: Math.round(timeSec),
              type: 'pass'
            });
            foundPass = true;
          }
        }
      }
      
      if (clips.length >= 6) break;
    }
    
    if (clips.length < 3) {
      return [
        { id: 'pass-demo', title: 'Passing Arc', desc: '3D Passing Trajectory Arc', time: 9, type: 'pass' },
        { id: 'tri-demo', title: 'Support Triangle', desc: 'Teammate Passing Network', time: 13, type: 'triangle' },
        { id: 'switch-demo', title: 'Flank Switch', desc: 'Long Curved Flank Switch', time: 38, type: 'pass' },
        { id: 'press-demo', title: 'Defensive Press', desc: 'Defensive Line & Free Man', time: 71, type: 'defense' }
      ];
    }
    
    return clips;
  }, [trackingData, getPitchDistance]);

  const playClip = useCallback((timeSec: number) => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = timeSec;
    void v.play().catch((err) => console.log('Autoplay blocked or failed:', err));
  }, [videoRef]);

  const getPlayerSpeed = (player: any, pIdx: number) => {
    if (player.speed_kmh !== undefined && player.speed_kmh !== null) {
      return player.speed_kmh;
    }
    
    if (!trackingData?.frames || !player.id) return 0;
    const currentFrameIdx = activeFrame?.frame_idx;
    if (currentFrameIdx === undefined) return 0;
    
    const lookback = 5;
    const targetIdx = currentFrameIdx - lookback;
    
    const prevFrame = trackingData.frames.find((f: any) => f.frame_idx === targetIdx);
    if (!prevFrame) return 0;
    
    const prevPlayer = prevFrame.players.find((p: any) => p.id === player.id);
    if (!prevPlayer || prevPlayer.x_pitch === undefined || prevPlayer.y_pitch === undefined) return 0;
    
    const dx = player.x_pitch - prevPlayer.x_pitch;
    const dy = player.y_pitch - prevPlayer.y_pitch;
    const dist = Math.hypot(dx, dy);
    
    const fps = trackingData.telemetry?.fps || 25;
    const timeSec = lookback / fps;
    const speedMs = dist / timeSec;
    const speedKmh = speedMs * 3.6;
    
    return isNaN(speedKmh) ? 0 : Math.min(36, speedKmh);
  };

  const getTeamColor = (teamId: string) => {
    if (teamId === 'team_0') return '#ef4444'; // Red
    if (teamId === 'team_1') return '#3b82f6'; // Blue
    return '#a1a1aa';
  };

  const syncFromVideo = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    const d = v.duration;
    setDuration(Number.isFinite(d) ? d : 0);
    setCurrentTime(Number.isFinite(v.currentTime) ? v.currentTime : 0);
    setIsPlaying(!v.paused);
  }, [videoRef]);

  useEffect(() => {
    const v = videoRef.current;
    if (!v || !videoSrc) {
      setCurrentTime(0);
      setDuration(0);
      setIsPlaying(false);
      return;
    }

    const onLoadedMetadata = (): void => syncFromVideo();
    const onTimeUpdate = (): void => syncFromVideo();
    const onSeeked = (): void => syncFromVideo();
    const onPlay = (): void => setIsPlaying(true);
    const onPause = (): void => setIsPlaying(false);
    const onEnded = (): void => {
      setIsPlaying(false);
      syncFromVideo();
    };

    v.addEventListener("loadedmetadata", onLoadedMetadata);
    v.addEventListener("durationchange", onLoadedMetadata);
    v.addEventListener("timeupdate", onTimeUpdate);
    v.addEventListener("seeked", onSeeked);
    v.addEventListener("play", onPlay);
    v.addEventListener("pause", onPause);
    v.addEventListener("ended", onEnded);

    if (v.readyState >= HTMLMediaElement.HAVE_METADATA) {
      syncFromVideo();
    }

    return () => {
      v.removeEventListener("loadedmetadata", onLoadedMetadata);
      v.removeEventListener("durationchange", onLoadedMetadata);
      v.removeEventListener("timeupdate", onTimeUpdate);
      v.removeEventListener("seeked", onSeeked);
      v.removeEventListener("play", onPlay);
      v.removeEventListener("pause", onPause);
      v.removeEventListener("ended", onEnded);
    };
  }, [videoRef, videoSrc, syncFromVideo]);

  const togglePlay = (): void => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) {
      void v.play();
    } else {
      v.pause();
    }
  };

  const onSeekInput = (e: ChangeEvent<HTMLInputElement>): void => {
    const v = videoRef.current;
    if (!v) return;
    const t = parseFloat(e.target.value);
    if (!Number.isFinite(t)) return;
    v.currentTime = t;
    setCurrentTime(t);
  };

  const onVideoAreaClick = (): void => {
    togglePlay();
  };

  const maxT = Number.isFinite(duration) && duration > 0 ? duration : 0;
  const rangeValue = Number.isFinite(currentTime) ? Math.min(Math.max(currentTime, 0), maxT || 0) : 0;

  return (
    <div className="flex h-full min-h-0 min-w-0 flex-1 flex-col gap-2">
      <div
        ref={containerRef}
        className={`flex flex-1 min-h-0 flex-col overflow-hidden bg-black shadow-2xl relative ${
          isFullscreen ? "w-screen h-screen rounded-none border-0" : "rounded-xl border border-gray-800"
        }`}
      >
        <div className="relative flex min-h-0 flex-1 flex-col">
          <div className="absolute left-3 top-3 z-10 flex gap-2">
            <span className="rounded bg-black/60 px-2 py-1 text-xs font-bold text-white backdrop-blur-md">
              LIVE FEED
            </span>
            {jobId && (
              <span className="rounded bg-blue-900/80 px-2 py-1 text-xs font-mono text-blue-100 backdrop-blur-md">
                {jobId.slice(0, 8)}…
              </span>
            )}
          </div>

          {videoSrc ? (
            <>
              <div
                className="relative min-h-0 flex-1 cursor-pointer bg-black flex items-center justify-center"
                onClick={onVideoAreaClick}
                onKeyDown={(e) => {
                  if (e.key === " " || e.key === "Enter") {
                    e.preventDefault();
                    togglePlay();
                  }
                }}
                role="button"
                tabIndex={0}
                aria-label="Play or pause video"
              >
                <video
                  ref={videoRef}
                  src={videoSrc}
                  className="pointer-events-none h-full w-full object-contain"
                  playsInline
                  preload="metadata"
                />
                
                {/* SVG Visual Overlays Layer */}
                {showOverlays && activeFrame && (
                  <svg
                    className="absolute pointer-events-none select-none z-10"
                    style={{
                      width: geometry.width,
                      height: geometry.height,
                      left: geometry.left,
                      top: geometry.top,
                    }}
                    viewBox={`0 0 ${geometry.width} ${geometry.height}`}
                  >
                    {/* SVG Filters and Definitions */}
                    <defs>
                      <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
                        <feDropShadow dx="0" dy="2" stdDeviation="3" floodOpacity="0.6" floodColor="#000000"/>
                      </filter>
                      <linearGradient id="pass-grad" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor="#ffffff" stopOpacity="0.4" />
                        <stop offset="50%" stopColor="#22c55e" stopOpacity="0.9" />
                        <stop offset="100%" stopColor="#22c55e" stopOpacity="1" />
                      </linearGradient>
                    </defs>

                    {/* A. Support Triangles (rendered below lines and players) */}
                    {tacticalTriangles.map((tri, tIdx) => (
                      <polygon
                        key={`support-tri-${tIdx}`}
                        points={tri.pointsStr}
                        fill={`${tri.color}22`}
                        stroke={tri.color}
                        strokeWidth={1.5}
                        strokeDasharray="4,3"
                        opacity={0.75}
                        style={{ filter: "url(#shadow)" }}
                      />
                    ))}

                    {/* B. Curved Passing Ball Trajectory Arrow */}
                    {passArrow && (
                      <g key="ball-pass-arrow" style={{ filter: "url(#shadow)" }}>
                        <path
                          d={passArrow.d}
                          fill="none"
                          stroke="rgba(0, 0, 0, 0.4)"
                          strokeWidth={5}
                          strokeLinecap="round"
                        />
                        <path
                          d={passArrow.d}
                          fill="none"
                          stroke="url(#pass-grad)"
                          strokeWidth={3}
                          strokeLinecap="round"
                        />
                        <polygon
                          points={passArrow.arrowPoints}
                          fill="#22c55e"
                          stroke="white"
                          strokeWidth={1}
                        />
                        <g transform={`translate(${passArrow.peak.x}, ${passArrow.peak.y})`}>
                          <rect
                            x={-28}
                            y={-14}
                            width={56}
                            height={15}
                            rx={4}
                            fill="rgba(10, 15, 10, 0.85)"
                            stroke="#22c55e"
                            strokeWidth={1}
                          />
                          <text
                            x={0}
                            y={-3}
                            textAnchor="middle"
                            fill="#ffffff"
                            fontSize="9px"
                            fontWeight="extrabold"
                          >
                            {`${passArrow.distance.toFixed(1)} m`}
                          </text>
                        </g>
                      </g>
                    )}

                    {/* 1. Connection Lines for Defenders (Back Four Line) */}
                    {(() => {
                      const { team0, team1 } = backLineDefenders;
                      const renders: React.ReactNode[] = [];
                      
                      [
                        { defenders: team0, color: '#ef4444', label: `${useAltNames && dictionary["team_0"] ? dictionary["team_0"] : "Red Team"} Line` },
                        { defenders: team1, color: '#3b82f6', label: `${useAltNames && dictionary["team_1"] ? dictionary["team_1"] : "Blue Team"} Line` }
                      ].forEach(({ defenders, color, label }, idx) => {
                        if (defenders.length < 2) return;
                        const pts = defenders.map(p => {
                          const x = p.x_canvas * geometry.scaleX;
                          const y = (p.bbox ? p.bbox[3] : (p.y_canvas + 15)) * geometry.scaleY;
                          return { x, y };
                        });
                        
                        const pointsStr = pts.map(p => `${p.x},${p.y}`).join(' ');
                        
                        // Calculate center for text label
                        const cx = pts.reduce((sum, p) => sum + p.x, 0) / pts.length;
                        const cy = pts.reduce((sum, p) => sum + p.y, 0) / pts.length;
                        
                        renders.push(
                          <g key={`backline-${idx}`}>
                            {/* Connection Line */}
                            <polyline
                              points={pointsStr}
                              fill="none"
                              stroke={color}
                              strokeWidth={2}
                              strokeDasharray="6,4"
                              opacity={0.8}
                            />
                            {/* Pulse highlights on joints */}
                            {pts.map((p, pIdx) => (
                              <circle
                                key={`joint-${pIdx}`}
                                cx={p.x}
                                cy={p.y}
                                r={4}
                                fill={color}
                                opacity={0.9}
                              />
                            ))}
                            {/* Label Background */}
                            <rect
                              x={cx - 40}
                              y={cy - 20}
                              width={80}
                              height={13}
                              rx={3}
                              fill="rgba(0, 0, 0, 0.75)"
                              stroke={color}
                              strokeWidth={1}
                            />
                            {/* Label Text */}
                            <text
                              x={cx}
                              y={cy - 10}
                              textAnchor="middle"
                              fill="white"
                              fontSize="8px"
                              fontWeight="bold"
                            >
                              {label}
                            </text>
                          </g>
                        );
                      });
                      return renders;
                    })()}

                    {/* 2. Circles and Speed Badges for each player */}
                    {filteredPlayers.map((p: any, pIdx: number) => {
                      if (p.x_canvas === undefined || p.y_canvas === undefined) return null;
                      
                      const x_screen = p.x_canvas * geometry.scaleX;
                      
                      // Calculate feet position: bottom center of bbox, or offset center
                      const y_feet = p.bbox ? p.bbox[3] * geometry.scaleY : (p.y_canvas + 15) * geometry.scaleY;
                      // Calculate head position: top center of bbox, or offset center
                      const y_head = p.bbox ? p.bbox[1] * geometry.scaleY : (p.y_canvas - 15) * geometry.scaleY;
                      
                      const color = getTeamColor(p.team_id || p.team);
                      const speed = getPlayerSpeed(p, pIdx);
                      
                      if (!p.team_id && !p.team) return null;
                      
                      const isDefender = backLineDefenders.team0.some((d: any) => d.id === p.id) || 
                                         backLineDefenders.team1.some((d: any) => d.id === p.id);
                                         
                      return (
                        <g key={`player-ann-${p.id || pIdx}`}>
                          {/* Perspective Ellipse under player feet */}
                          <ellipse
                            cx={x_screen}
                            cy={y_feet}
                            rx={isDefender ? 22 : 18}
                            ry={isDefender ? 7.5 : 6}
                            stroke={color}
                            strokeWidth={2}
                            fill={`${color}22`}
                            filter="drop-shadow(0 0 3px rgba(0,0,0,0.5))"
                          />
                          
                          {/* Connecting line to head tag */}
                          <line
                            x1={x_screen}
                            y1={y_feet}
                            x2={x_screen}
                            y2={y_head}
                            stroke={color}
                            strokeWidth={1}
                            strokeDasharray="2,2"
                            opacity={0.4}
                          />
                          
                          {/* Player ID/Role Pill */}
                          <g>
                            {/* Background Tag Pill */}
                            <rect
                              x={x_screen - 26}
                              y={y_head - 26}
                              width={52}
                              height={11}
                              rx={2}
                              fill={color}
                              opacity={0.9}
                            />
                            {/* Text */}
                            <text
                              x={x_screen}
                              y={y_head - 18}
                              textAnchor="middle"
                              fill="white"
                              fontSize="8px"
                              fontWeight="bold"
                            >
                              {useAltNames && dictionary[`P${p.id}`] ? dictionary[`P${p.id}`] : `PLAYER ${p.id}`}
                            </text>
                            
                            {/* Speed Label */}
                            {speed > 0 && (
                              <text
                                x={x_screen}
                                y={y_head - 6}
                                textAnchor="middle"
                                fill="#ffffff"
                                fontSize="9px"
                                fontWeight="bold"
                                filter="drop-shadow(0px 1px 2px rgba(0,0,0,0.95))"
                              >
                                {`${speed.toFixed(1)} km/h`}
                              </text>
                            )}
                          </g>
                        </g>
                      );
                    })}
                  </svg>
                )}

                <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/20">
                  {!isPlaying && (
                    <div className="flex h-16 w-16 items-center justify-center rounded-full bg-white/20 backdrop-blur-sm transition-transform hover:scale-110">
                      <svg className="ml-1 h-8 w-8 text-white" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M8 5v14l11-7z" />
                      </svg>
                    </div>
                  )}
                </div>
              </div>

              <div
                className="flex shrink-0 items-center gap-2 border-t border-gray-800 bg-black/90 px-2 py-2"
                onClick={(e) => e.stopPropagation()}
                onKeyDown={(e) => e.stopPropagation()}
              >
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    togglePlay();
                  }}
                  className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-white/10 text-white transition hover:bg-white/20"
                  aria-label={isPlaying ? "Pause" : "Play"}
                >
                  {isPlaying ? (
                    <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
                    </svg>
                  ) : (
                    <svg className="ml-0.5 h-5 w-5" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M8 5v14l11-7z" />
                    </svg>
                  )}
                </button>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    setShowOverlays(prev => !prev);
                  }}
                  className={`flex h-9 px-2.5 shrink-0 items-center justify-center gap-1.5 rounded-md text-xs font-bold transition ${
                    showOverlays 
                      ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 hover:bg-emerald-500/30" 
                      : "bg-white/10 text-gray-400 hover:bg-white/20"
                  }`}
                  aria-label="Toggle Tactical Overlays"
                >
                  <svg className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                  </svg>
                  <span>Overlays</span>
                </button>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    toggleFullscreen();
                  }}
                  className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-white/10 text-white transition hover:bg-white/20"
                  aria-label={isFullscreen ? "Exit Fullscreen" : "Fullscreen"}
                >
                  {isFullscreen ? (
                    <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M9 9L3 3m0 0l2 2m-2-2h5M15 9l6-6m0 0l-2 2m2-2h-5M9 15l-6 6m0 0l2-2m-2 2h5M15 15l6 6m0 0l-2-2m2-2h-5" />
                    </svg>
                  ) : (
                    <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5m-7 11H4m0 0v-4m0 4l5-5m11 5v-4m0 4h-4m4 0l-5-5" />
                    </svg>
                  )}
                </button>
                <input
                  type="range"
                  min={0}
                  max={maxT > 0 ? maxT : 0}
                  step="any"
                  value={maxT > 0 ? rangeValue : 0}
                  onChange={onSeekInput}
                  className="h-2 min-w-0 flex-1 cursor-pointer accent-emerald-500"
                  aria-label="Seek video"
                  disabled={maxT <= 0}
                />
                <span className="shrink-0 font-mono text-xs tabular-nums text-gray-300">
                  {formatClock(currentTime)} / {formatClock(duration)}
                </span>
              </div>
              {videoSrc && (
                <div className="border-t border-gray-900 bg-[#080d08]/85 px-4 py-3">
                  <h4 className="text-[9px] font-bold font-mono uppercase tracking-[0.2em] text-gray-500 mb-2">
                    Tactical Action Clips
                  </h4>
                  <div className="flex gap-3 overflow-x-auto pb-1 [&::-webkit-scrollbar]:h-1 [&::-webkit-scrollbar-thumb]:bg-gray-800 scrollbar-thin scrollbar-thumb-gray-800">
                    {detectedClips.map((clip) => {
                      return (
                        <button
                          key={clip.id}
                          type="button"
                          onClick={() => playClip(clip.time)}
                          className="flex min-w-[210px] items-center gap-3 rounded-xl border border-gray-850 bg-[#060a06]/90 p-2.5 text-left transition hover:scale-[1.02] hover:border-emerald-500/40 hover:bg-[#111a12]/50 active:scale-[0.98] group"
                        >
                          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-emerald-500/10 text-emerald-400 group-hover:bg-emerald-500/20 transition-colors">
                            {clip.type === "pass" ? (
                              <svg className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" d="M13 5l7 7-7 7M5 5l7 7-7 7" />
                              </svg>
                            ) : clip.type === "triangle" ? (
                              <svg className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" d="M4 19L12 5l8 14H4z" />
                              </svg>
                            ) : (
                              <svg className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                              </svg>
                            )}
                          </div>
                          <div className="min-w-0 flex-1">
                            <div className="flex items-center justify-between gap-1">
                              <span className="truncate text-xs font-bold text-white group-hover:text-emerald-400 transition-colors">
                                {clip.title}
                              </span>
                              <span className="shrink-0 font-mono text-[9px] font-bold text-gray-500">
                                {formatClock(clip.time)}
                              </span>
                            </div>
                            <span className="block truncate text-[10px] text-gray-500">
                              {clip.desc}
                            </span>
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="flex flex-1 flex-col items-center justify-center gap-4 p-6 text-center text-gray-500">
              <div className="rounded-full bg-gray-900 p-6">
                <svg className="h-12 w-12 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1}
                    d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                  />
                </svg>
              </div>
              <div>
                <p className="text-sm font-medium text-gray-400">No video loaded</p>
                <p className="mt-1 text-xs">Upload a clip to begin analysis</p>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="flex items-center justify-between px-1">
        <span className="text-[10px] font-bold uppercase tracking-wider text-gray-500">Source</span>
        <div className="flex items-center gap-2">
          <span
            className={`text-[10px] font-bold uppercase ${
              status === "completed" ? "text-emerald-500" : "text-gray-600"
            }`}
          >
            {status}
          </span>
          <button
            type="button"
            onClick={onDownload}
            disabled={status !== "completed"}
            className="rounded border border-gray-800 bg-gray-900 px-2 py-1 text-[10px] font-bold uppercase text-gray-300 transition hover:bg-gray-800 disabled:opacity-50"
          >
            Download JSON
          </button>
        </div>
      </div>
    </div>
  );
}

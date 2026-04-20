import { useState, useCallback, useRef, useEffect } from 'react';
import { resolveApiEndpoints } from '@/lib/apiBase';
import { debugSessionLog } from '@/lib/debugSessionLog';

export type ProgressStep = string;

/** Milestone hints for UI only; backend `current_step` strings may differ. */
export const STEPS: ProgressStep[] = [
  'Tracking Players',
  'Spatial Math',
  'Rule Engine',
  'Synthesizing Advice',
];

type JobWsPayload = {
  status?: string;
  current_step?: string;
  error?: string;
  error_code?: string | null;
};

export function useWebSocketProgress() {
  const [currentStep, setCurrentStep] = useState<ProgressStep>('Pending');
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [connectionState, setConnectionState] = useState<
    'idle' | 'connecting' | 'open' | 'closed' | 'error'
  >('idle');
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);
  const stepStartedAtRef = useRef<number | null>(null);
  const terminalStatusSeenRef = useRef(false);
  const isProcessingRef = useRef(false);

  useEffect(() => {
    isProcessingRef.current = isProcessing;
  }, [isProcessing]);

  useEffect(() => {
    if (!isProcessing) return;
    const tick = window.setInterval(() => {
      if (stepStartedAtRef.current == null) {
        return;
      }
      const ms = Date.now() - stepStartedAtRef.current;
      setElapsedSeconds(Math.max(0, Math.floor(ms / 1000)));
    }, 1000);
    return () => window.clearInterval(tick);
  }, [isProcessing]);

  const startTracking = useCallback((jobId: string) => {
    setIsProcessing(true);
    setConnectionState('connecting');
    setError(null);
    setCurrentStep('Pending');
    setElapsedSeconds(0);
    stepStartedAtRef.current = Date.now();
    terminalStatusSeenRef.current = false;

    const { wsBase, host, source } = resolveApiEndpoints();
    const wsUrl = `${wsBase}/ws/jobs/${jobId}`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    // #region agent log
    debugSessionLog({
      sessionId: 'bb63ae',
      hypothesisId: 'H2-H3',
      location: 'useWebSocketProgress.ts:startTracking',
      message: 'WS open url',
      data: {
        wsHost: host,
        apiBaseSource: source,
        jobIdPrefix: jobId.slice(0, 8),
      },
    });
    // #endregion

    ws.onopen = () => {
      setConnectionState('open');
      debugSessionLog({
        sessionId: 'bb63ae',
        hypothesisId: 'H2-H3',
        location: 'useWebSocketProgress.ts:onopen',
        message: 'WS connected',
        data: { wsHost: host, jobIdPrefix: jobId.slice(0, 8) },
      });
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as JobWsPayload;
        // #region agent log
        debugSessionLog({
          sessionId: 'bb63ae',
          hypothesisId: 'H2',
          location: 'useWebSocketProgress.ts:onmessage',
          message: 'WS payload',
          data: {
            status: data.status,
            current_step: data.current_step,
            hasError: !!data.error,
            error_code: data.error_code ?? null,
          },
        });
        // #endregion
        if (data.current_step != null && data.current_step !== '') {
          setCurrentStep((prev) => {
            if (prev !== data.current_step) {
              stepStartedAtRef.current = Date.now();
              setElapsedSeconds(0);
            }
            return data.current_step as string;
          });
        }
        const status = data.status;
        if (status === 'done') {
          terminalStatusSeenRef.current = true;
          setCurrentStep('Completed');
          setIsProcessing(false);
          setConnectionState('closed');
          ws.close();
          return;
        }
        if (status === 'error') {
          terminalStatusSeenRef.current = true;
          const detail =
            typeof data.error === 'string' && data.error.length > 0
              ? data.error
              : 'Job failed';
          const codePrefix = data.error_code ? `[${data.error_code}] ` : '';
          setError(`${codePrefix}${detail}`);
          setIsProcessing(false);
          setConnectionState('error');
          ws.close();
        }
      } catch (e) {
        console.error('Failed to parse WS message', e);
      }
    };

    ws.onerror = (e) => {
      console.error('WebSocket Error', e);
      // #region agent log
      debugSessionLog({
        sessionId: 'bb63ae',
        hypothesisId: 'H2-H3',
        location: 'useWebSocketProgress.ts:onerror',
        message: 'WS error',
        data: { jobIdPrefix: jobId.slice(0, 8), wsHost: host },
      });
      // #endregion
      setError(
        `Connection error while streaming job progress (host: ${host}). Check API base URL and backend server.`,
      );
      setIsProcessing(false);
      setConnectionState('error');
    };

    ws.onclose = () => {
      if (isProcessingRef.current && !terminalStatusSeenRef.current) {
        setError(
          'Progress stream closed before completion. Check backend logs and API URL wiring.',
        );
      }
      setIsProcessing(false);
      setConnectionState((prev) => (prev === 'error' ? prev : 'closed'));
    };

  }, []);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
    }
    setIsProcessing(false);
    setConnectionState('closed');
  }, []);

  return {
    currentStep,
    isProcessing,
    error,
    connectionState,
    elapsedSeconds,
    startTracking,
    disconnect,
  };
}

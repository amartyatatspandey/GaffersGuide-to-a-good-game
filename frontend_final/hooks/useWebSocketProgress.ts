import { useState, useCallback, useRef } from 'react';
import { getWsBaseUrl } from '@/lib/apiBase';
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
};

export function useWebSocketProgress() {
  const [currentStep, setCurrentStep] = useState<ProgressStep>('Pending');
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const startTracking = useCallback((jobId: string) => {
    setIsProcessing(true);
    setError(null);
    setCurrentStep('Pending');

    const wsUrl = `${getWsBaseUrl()}/ws/jobs/${jobId}`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    // #region agent log
    debugSessionLog({
      sessionId: 'bb63ae',
      hypothesisId: 'H2-H3',
      location: 'useWebSocketProgress.ts:startTracking',
      message: 'WS open url',
      data: {
        wsHost: (() => {
          try {
            return new URL(wsUrl).host;
          } catch {
            return 'bad-ws-url';
          }
        })(),
        jobIdPrefix: jobId.slice(0, 8),
      },
    });
    // #endregion

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
          },
        });
        // #endregion
        if (data.current_step != null && data.current_step !== '') {
          setCurrentStep(data.current_step);
        }
        const status = data.status;
        if (status === 'done') {
          setCurrentStep('Completed');
          setIsProcessing(false);
          ws.close();
          return;
        }
        if (status === 'error') {
          const detail =
            typeof data.error === 'string' && data.error.length > 0
              ? data.error
              : 'Job failed';
          setError(detail);
          setIsProcessing(false);
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
        data: { jobIdPrefix: jobId.slice(0, 8) },
      });
      // #endregion
      setError('Connection Error');
      setIsProcessing(false);
    };

    ws.onclose = () => {
      setIsProcessing(false);
    };

  }, []);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
    }
    setIsProcessing(false);
  }, []);

  return { currentStep, isProcessing, error, startTracking, disconnect };
}

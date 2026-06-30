import { useState, useCallback, useRef, useEffect } from 'react';
import { getWsBaseUrl, getAuthHeaders } from '@/lib/apiBase';
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
  const pollingTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const isUnmounted = useRef(false);

  useEffect(() => {
    return () => {
      isUnmounted.current = true;
      if (pollingTimeoutRef.current) {
        clearTimeout(pollingTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const startTracking = useCallback((jobId: string) => {
    setIsProcessing(true);
    setError(null);
    setCurrentStep('Pending');
    isUnmounted.current = false;

    const connectWs = (reconnectAttempts = 0) => {
      if (isUnmounted.current) return;

      const supabaseToken = typeof window !== 'undefined' ? localStorage.getItem("gaffer-supabase-token") : null;
      const apiKey = process.env.NEXT_PUBLIC_API_KEY;
      
      let wsUrl = `${getWsBaseUrl()}/ws/jobs/${jobId}`;
      if (supabaseToken) {
        wsUrl += `?token=${encodeURIComponent(supabaseToken)}`;
      } else if (apiKey) {
        wsUrl += `?api_key=${encodeURIComponent(apiKey)}`;
      }

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
      };

      ws.onclose = () => {
        if (isUnmounted.current) return;
        
        setCurrentStep((prev) => {
          if (prev === 'Completed') {
            setIsProcessing(false);
            return prev;
          }
          
          // Reconnect attempt if we haven't reached max retries
          if (reconnectAttempts < 3) {
            setTimeout(() => connectWs(reconnectAttempts + 1), 2000);
            return prev;
          }

          // Fallback to polling
          const pollStatus = async () => {
            if (isUnmounted.current) return;
            try {
              const res = await fetch(`${getWsBaseUrl().replace('ws://', 'http://').replace('wss://', 'https://')}/api/v1/jobs/${jobId}/artifacts`, { headers: getAuthHeaders() });
              if (res.ok) {
                const data = await res.json();
                if (data.status === 'done') {
                  setCurrentStep('Completed');
                  setIsProcessing(false);
                  return;
                } else if (data.status === 'error') {
                  setError('Job failed during processing.');
                  setIsProcessing(false);
                  return;
                }
              } else if (res.status === 404) {
                setError('job_not_found');
                setIsProcessing(false);
                return;
              }
            } catch (e) {
              // Ignore fetch errors, try again next tick
            }
            if (!isUnmounted.current) {
              pollingTimeoutRef.current = setTimeout(pollStatus, 2000);
            }
          };
          
          pollStatus();
          return prev;
        });
      };
    };

    connectWs();

  }, []);

  const disconnect = useCallback(() => {
    isUnmounted.current = true;
    if (wsRef.current) {
      wsRef.current.close();
    }
    if (pollingTimeoutRef.current) {
      clearTimeout(pollingTimeoutRef.current);
    }
    setIsProcessing(false);
  }, []);

  return { currentStep, isProcessing, error, startTracking, disconnect };
}

import { useState, useCallback, useRef } from 'react';
import { getWsBaseUrl } from '../lib/apiBase';

export type ProgressStep = string;

// Common steps matching backend's potential output
export const STEPS: ProgressStep[] = [
  'Tracking Players',
  'Spatial Math',
  'Rule Engine',
  'Synthesizing Advice'
];

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

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.current_step) {
          setCurrentStep(data.current_step);
        }
        if (data.status === 'Completed' || data.status === 'Failed' || data.type === 'error') {
          if (data.status === 'Failed' || data.type === 'error') {
             setError(data.message || 'Job Failed');
          } else {
             setCurrentStep('Completed');
          }
          setIsProcessing(false);
          ws.close();
        }
      } catch (e) {
        console.error('Failed to parse WS message', e);
      }
    };

    ws.onerror = (e) => {
      console.error('WebSocket Error', e);
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

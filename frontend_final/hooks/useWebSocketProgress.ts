import { useState, useEffect } from 'react';

export type ProgressStep = 
  | 'Pending'
  | 'Tracking Players'
  | 'Spatial Math'
  | 'Rule Engine'
  | 'Synthesizing Advice'
  | 'Completed';

export const STEPS: ProgressStep[] = [
  'Tracking Players',
  'Spatial Math',
  'Rule Engine',
  'Synthesizing Advice'
];

export function useWebSocketProgress() {
  const [currentStep, setCurrentStep] = useState<ProgressStep>('Pending');
  const [isProcessing, setIsProcessing] = useState(false);

  const startProcessing = () => {
    setIsProcessing(true);
    setCurrentStep('Tracking Players');
  };

  useEffect(() => {
    if (!isProcessing) return;

    const timer1 = setTimeout(() => setCurrentStep('Spatial Math'), 2000);
    const timer2 = setTimeout(() => setCurrentStep('Rule Engine'), 4000);
    const timer3 = setTimeout(() => setCurrentStep('Synthesizing Advice'), 6000);
    const timer4 = setTimeout(() => {
      setCurrentStep('Completed');
      setIsProcessing(false);
    }, 8000);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
      clearTimeout(timer4);
    };
  }, [isProcessing]);

  return { currentStep, isProcessing, startProcessing };
}

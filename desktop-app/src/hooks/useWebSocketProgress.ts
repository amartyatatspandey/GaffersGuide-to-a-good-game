import { useEffect, useRef, useState } from "react";

import {
  type CoachAdviceResponse,
  type JobArtifactsResponse,
  type JobProgressMessage,
  createBetaJob,
  getBetaArtifacts,
  getCoachAdvice,
  subscribeJobProgress,
} from "@/lib/api";

export type ProgressStep =
  | "Pending"
  | "Tracking Players"
  | "Spatial Math"
  | "Rule Engine"
  | "Synthesizing Advice"
  | "Completed"
  | "Error";

export const STEPS: ProgressStep[] = [
  "Tracking Players",
  "Spatial Math",
  "Rule Engine",
  "Synthesizing Advice",
];

interface UseWebSocketProgressResult {
  currentStep: ProgressStep;
  isProcessing: boolean;
  errorMessage: string | null;
  jobId: string | null;
  artifacts: JobArtifactsResponse | null;
  advice: CoachAdviceResponse | null;
  startProcessing: (file: File) => Promise<void>;
}

function normalizeStep(step: string): ProgressStep {
  if (step === "Completed") {
    return "Completed";
  }
  if (step === "Error") {
    return "Error";
  }
  if (STEPS.includes(step as ProgressStep)) {
    return step as ProgressStep;
  }
  return "Pending";
}

export function useWebSocketProgress(): UseWebSocketProgressResult {
  const [currentStep, setCurrentStep] = useState<ProgressStep>("Pending");
  const [isProcessing, setIsProcessing] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [artifacts, setArtifacts] = useState<JobArtifactsResponse | null>(null);
  const [advice, setAdvice] = useState<CoachAdviceResponse | null>(null);
  const unsubscribeRef = useRef<(() => void) | null>(null);

  useEffect(
    () => () => {
      if (unsubscribeRef.current) {
        unsubscribeRef.current();
      }
    },
    [],
  );

  const startProcessing = async (file: File): Promise<void> => {
    setIsProcessing(true);
    setErrorMessage(null);
    setCurrentStep("Pending");
    setArtifacts(null);
    setAdvice(null);

    try {
      const job = await createBetaJob(file);
      setJobId(job.job_id);
      setCurrentStep("Tracking Players");

      if (unsubscribeRef.current) {
        unsubscribeRef.current();
      }

      unsubscribeRef.current = subscribeJobProgress(
        job.job_id,
        async (message: JobProgressMessage) => {
          if (message.error) {
            setErrorMessage(message.error);
          }
          setCurrentStep(normalizeStep(message.current_step));

          if (message.status === "done") {
            setCurrentStep("Completed");
            setIsProcessing(false);
            const [artifactPayload, advicePayload] = await Promise.all([
              getBetaArtifacts(job.job_id),
              getCoachAdvice(job.job_id),
            ]);
            setArtifacts(artifactPayload);
            setAdvice(advicePayload);
          } else if (message.status === "error") {
            setCurrentStep("Error");
            setIsProcessing(false);
            setErrorMessage(message.error ?? "Pipeline failed.");
          }
        },
        (error: Error) => {
          setCurrentStep("Error");
          setIsProcessing(false);
          setErrorMessage(error.message);
        },
      );
    } catch (error) {
      setIsProcessing(false);
      setCurrentStep("Error");
      setErrorMessage(error instanceof Error ? error.message : "Job creation failed.");
    }
  };

  return {
    currentStep,
    isProcessing,
    errorMessage,
    jobId,
    artifacts,
    advice,
    startProcessing,
  };
}

import { useEffect, useRef, useState } from "react";

import {
  type CoachAdviceResponse,
  type CreateBetaJobOptions,
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

async function sleep(ms: number): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

function isRetryableNotReadyError(error: unknown): boolean {
  if (!(error instanceof Error)) {
    return false;
  }
  return error.message.includes("(425)") || error.message.toLowerCase().includes("not ready");
}

async function fetchWithRetry<T>(
  fetcher: () => Promise<T>,
  attempts = 4,
  delayMs = 700,
): Promise<T> {
  let lastError: unknown;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      return await fetcher();
    } catch (error) {
      lastError = error;
      if (!isRetryableNotReadyError(error) || attempt === attempts) {
        throw error;
      }
      await sleep(delayMs * attempt);
    }
  }
  throw lastError;
}

interface UseWebSocketProgressResult {
  currentStep: ProgressStep;
  isProcessing: boolean;
  errorMessage: string | null;
  jobId: string | null;
  artifacts: JobArtifactsResponse | null;
  advice: CoachAdviceResponse | null;
  startProcessing: (file: File, options?: CreateBetaJobOptions) => Promise<void>;
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

  const startProcessing = async (file: File, options?: CreateBetaJobOptions): Promise<void> => {
    setIsProcessing(true);
    setErrorMessage(null);
    setCurrentStep("Pending");
    setArtifacts(null);
    setAdvice(null);

    try {
      const job = await createBetaJob(file, options);
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
            try {
              const [artifactPayload, advicePayload] = await Promise.all([
                fetchWithRetry(() => getBetaArtifacts(job.job_id)),
                fetchWithRetry(() =>
                  getCoachAdvice(job.job_id, { llmEngine: options?.llmEngine }),
                ),
              ]);
              setArtifacts(artifactPayload);
              setAdvice(advicePayload);
              if (advicePayload.advice_items.length === 0) {
                setErrorMessage(
                  "No tactical insights crossed rule thresholds for this match. Try a longer clip or adjust analysis settings.",
                );
              }
            } catch (error) {
              setErrorMessage(
                error instanceof Error
                  ? error.message
                  : "Artifacts/advice fetch failed after job completion.",
              );
            }
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

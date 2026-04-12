import { useCallback, useEffect, useState } from "react";

import { checkBackendHealth } from "@/lib/api";

export type BackendHealthStatus = "checking" | "ok" | "unreachable";

export interface UseBackendHealthResult {
  status: BackendHealthStatus;
  recheck: () => void;
}

export function useBackendHealth(pollMs = 0): UseBackendHealthResult {
  const [status, setStatus] = useState<BackendHealthStatus>("checking");

  const check = useCallback(async (): Promise<void> => {
    setStatus("checking");
    const result = await checkBackendHealth();
    setStatus(result.reachable ? "ok" : "unreachable");
  }, []);

  useEffect(() => {
    void check();
    if (!pollMs) return;
    const id = setInterval(() => void check(), pollMs);
    return () => clearInterval(id);
  }, [check, pollMs]);

  return { status, recheck: () => void check() };
}

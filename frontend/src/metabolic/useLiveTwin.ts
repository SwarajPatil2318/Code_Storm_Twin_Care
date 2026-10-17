import { useEffect, useRef, useState } from "react";
import type { Pack, State, Trajectory } from "./types";
import { releaseDelay, type DemoSpeed, type Presentation } from "./story";

export type LiveResult = {
  session_id: string;
  state: State;
  index: number;
  complete: boolean;
  presentation: Presentation;
  observed_history: { timestamp: string; glucose_mg_dl: number }[];
  update: {
    trace_id: string;
    revision: number;
    mode: string;
    model_version: string;
    risk_model_version?: string;
    risk_sha256?: string;
    model_hashes: Record<string, string>;
    inference_timestamp: string;
    stream_label: string;
    pipeline: string[];
    timings_ms: Record<string, number>;
    input: {
      current_glucose: number;
      usable_cgm_observations: number;
      history_window_minutes: number;
      personal_prefix_observations: number;
      personal_baseline_available: boolean;
      recent_slope: number | null;
      event_time: string;
      receipt_time: string;
      meal_context: unknown;
      static_clinical_context_available: boolean;
    };
  };
  forecast_history: {
    prediction_time: string;
    model_version: string;
    risk_model_version?: string;
    risk_sha256?: string;
    forecast: Trajectory;
    spike_probability: number;
    peak: State["predicted_peak"];
    trace_id: string;
    comparisons: {
      horizon_minutes: number;
      predicted: number;
      observed: number;
      absolute_error_mg_dl: number;
    }[];
  }[];
  event_log: {
    trace_id: string;
    timestamp: string;
    glucose_mg_dl: number;
    stages: string[];
    timings_ms: Record<string, number>;
  }[];
};
async function post<T>(action: string, body: unknown): Promise<T> {
  const response = await fetch("/metabolic-api/metabolic/live/" + action, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(15000),
  });
  if (!response.ok)
    throw new Error("Live inference unavailable — using verified replay");
  return response.json();
}

export function useLiveTwin({
  id,
  packs,
  active,
  index,
  onIndex,
  onCase,
}: {
  id: string;
  packs: Record<string, Pack>;
  active: boolean;
  index: number;
  onIndex: (value: number) => void;
  onCase: (value: string) => void;
}) {
  const [result, setResult] = useState<LiveResult | null>(null);
  const [running, setRunning] = useState(false),
    [busy, setBusy] = useState(false);
  const [speed, setSpeed] = useState<DemoSpeed>("1"),
    [mode, setMode] = useState("HISTORICAL / DEMO REPLAY");
  const [warning, setWarning] = useState("");
  const current = useRef<LiveResult | null>(null),
    generation = useRef(0);
  const pendingJury = useRef<{ play: boolean } | null>(null),
    juryStart = useRef(24);
  const caseRef = useRef(id),
    packsRef = useRef(packs);
  caseRef.current = id;
  packsRef.current = packs;
  const dispose = () => {
    const old = current.current;
    current.current = null;
    if (old) void post("close", { session_id: old.session_id }).catch(() => {});
  };
  const fallback = (index: number) => {
    dispose();
    setResult(null);
    setRunning(false);
    setBusy(false);
    setMode("VERIFIED REPLAY");
    setWarning("Live inference unavailable — using verified replay");
    onIndex(index);
  };
  const initialize = async (
    index: number,
    selectedMode: string,
    play = false,
  ) => {
    const token = ++generation.current;
    const caseId = caseRef.current;
    dispose();
    setRunning(false);
    setBusy(true);
    setWarning("");
    setMode(selectedMode);
    setResult(null);
    onIndex(index);
    try {
      const next = await post<LiveResult>("session", {
        case_id: caseId,
        start_index: index,
        mode: selectedMode,
      });
      if (generation.current !== token) {
        void post("close", { session_id: next.session_id }).catch(() => {});
        return;
      }
      current.current = next;
      setResult(next);
      onIndex(next.index);
      setRunning(play && !next.complete);
    } catch {
      if (generation.current === token) fallback(index);
    } finally {
      if (generation.current === token) setBusy(false);
    }
  };
  const advance = async () => {
    const token = generation.current,
      previous = current.current;
    if (!previous || previous.complete || busy) return;
    setBusy(true);
    try {
      const next = await post<LiveResult>("advance", {
        session_id: previous.session_id,
        expected_revision: previous.update.revision,
      });
      if (generation.current !== token) return;
      current.current = next;
      setResult(next);
      onIndex(next.index);
      if (next.complete) setRunning(false);
    } catch {
      if (generation.current === token) fallback(previous.index);
    } finally {
      if (generation.current === token) setBusy(false);
    }
  };
  useEffect(() => {
    if (!id || !packs[id]) return;
    const jury = pendingJury.current;
    pendingJury.current = null;
    void initialize(
      jury ? juryStart.current : packs[id].frames.length - 1,
      jury ? "LIVE TWIN" : "HISTORICAL / DEMO REPLAY",
      jury?.play ?? false,
    );
    return () => {
      ++generation.current;
      dispose();
    };
  }, [id, packs]);
  useEffect(() => {
    if (!active) setRunning(false);
  }, [active]);
  useEffect(() => {
    const release = () => {
      if (current.current)
        navigator.sendBeacon(
          "/metabolic-api/metabolic/live/close",
          new Blob(
            [JSON.stringify({ session_id: current.current.session_id })],
            { type: "application/json" },
          ),
        );
    };
    window.addEventListener("pagehide", release);
    return () => window.removeEventListener("pagehide", release);
  }, []);
  useEffect(() => {
    if (!running || !active || busy || result?.complete) return;
    if (mode === "VERIFIED REPLAY") {
      const frames = packsRef.current[caseRef.current]?.frames;
      if (!frames || index >= frames.length - 1) {
        setRunning(false);
        return;
      }
      const timer = setTimeout(
        () => onIndex(index + 1),
        releaseDelay(speed, frames.length - 1),
      );
      return () => clearTimeout(timer);
    }
    const timer = setTimeout(
      () => void advance(),
      releaseDelay(speed, packsRef.current[caseRef.current].frames.length - 1),
    );
    return () => clearTimeout(timer);
  }, [running, active, busy, result, speed, mode, index]);
  const start = () => {
    if (running) {
      setRunning(false);
      return;
    }
    if (mode === "LIVE TWIN" && current.current && !current.current.complete)
      setRunning(true);
    else void initialize(0, "LIVE TWIN", true);
  };
  const replay = (index: number) => {
    if (running) {
      setRunning(false);
      return;
    }
    if (mode === "VERIFIED REPLAY") {
      if (index >= packsRef.current[caseRef.current].frames.length - 1)
        onIndex(0);
      setRunning(true);
      return;
    }
    if (
      current.current &&
      !current.current.complete &&
      mode === "HISTORICAL / DEMO REPLAY"
    )
      setRunning(true);
    else
      void initialize(
        index >= packsRef.current[caseRef.current].frames.length - 1
          ? 0
          : index,
        "HISTORICAL / DEMO REPLAY",
        true,
      );
  };
  const reset = () => void initialize(0, "LIVE TWIN", false);
  const jury = (play = true) => {
    juryStart.current = 0;
    setSpeed("demo");
    if (caseRef.current !== "predicted_spike") {
      pendingJury.current = { play };
      onCase("predicted_spike");
    } else void initialize(juryStart.current, "LIVE TWIN", play);
  };
  return {
    result,
    running,
    busy,
    mode,
    warning,
    speed,
    setSpeed,
    start,
    reset,
    jury,
    replay,
    pause: () => setRunning(false),
    seek: (index: number) =>
      void initialize(index, "HISTORICAL / DEMO REPLAY", false),
  };
}

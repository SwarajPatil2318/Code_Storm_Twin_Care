import {useReceivedSources} from './platform/useReceivedSources';
import { Activity, Database, Gauge, Scale, Watch } from "lucide-react";
import type { Observation, Signal } from "./types";
import { sourceReceipt } from "./sourceTiming";

const sources = [
  {
    name: "Smartwatch",
    subtitle: "HR · HRV · Activity",
    icon: Watch,
    signals: ["heart_rate", "hrv_rmssd", "steps", "sleep_hours"] as Signal[],
    view: "Rhythm",
    metric: "heart_rate" as Signal,
  },
  {
    name: "BP cuff",
    subtitle: "Blood pressure",
    icon: Gauge,
    signals: ["systolic_bp", "diastolic_bp"] as Signal[],
    view: "Fluid & pressure",
    metric: "systolic_bp" as Signal,
  },
  {
    name: "Smart scale",
    subtitle: "Weight",
    icon: Scale,
    signals: ["weight"] as Signal[],
    view: "Fluid & pressure",
    metric: "weight" as Signal,
  },
  {
    name: "Pulse oximeter",
    subtitle: "SpO₂",
    icon: Activity,
    signals: ["spo2"] as Signal[],
    view: "Recovery",
    metric: "spo2" as Signal,
  },
  {
    name: "EHR / FHIR",
    subtitle: "Diagnoses · meds · history",
    icon: Database,
    signals: [] as Signal[],
    view: "Clinical Details",
    metric: undefined,
  },
];
export function Sources({
  observations,
  cutoff,
  onSelect,
}: {
  observations: Observation[];
  cutoff: string;
  onSelect: (view: string, metric?: Signal) => void;
}) {
  const received=useReceivedSources(observations,cutoff);
  return (
    <section
      className="sources-strip"
      aria-label="Connected sources"
    >
      <div className="sources-heading">
        <h2>Connected sources</h2>

      </div>
      <div className="sources-row">
        {sources.map(
          ({ name, subtitle, icon: Icon, signals, view, metric }) => {
            const receipt = signals.length
              ? sourceReceipt(observations, signals, cutoff)
              : {
                  label: "FHIR context loaded",
                  detail:
                    "Clinical history available.",
                  available: true,
                };
            return (
              <button
                className={`source-chip ${received.includes(name)?"tc-received":""}`}
                key={name}
                title={receipt.detail}
                onClick={() => onSelect(view, metric)}
              >
                <span className="source-icon">
                  <Icon size={19} />
                </span>
                <span className="source-copy">
                  <strong>{name === "EHR / FHIR" ? "Medical records" : name === "Smartwatch" ? "Wearable" : name}</strong>
                  <small>{subtitle}</small>
                  <span
                    className={`source-status ${receipt.available ? "available" : "missing"}`}
                  >
                    <i />
                    {receipt.label}
                  </span>
                </span>
              </button>
            );
          },
        )}
      </div>
    </section>
  );
}

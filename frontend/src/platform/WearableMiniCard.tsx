import type { CSSProperties, ReactNode } from 'react';
import { Heart, Footprints, Activity, Moon, Bell, Check } from 'lucide-react';
import './wearable.css';

export type WearableReading = {
  label: string;
  value: number | string;
  unit?: string;
  region?: string;
  detail?: ReactNode;
  testId?: string;
};

/** Shared presentation only: each Twin supplies its existing received values. */
export function WearableMiniCard({mode, timestamp, heartRate, readings = [], received = false,
  onRegion, onEmphasis, heartRateDetail, notification, provenanceNote}: {
  mode: 'metabolic' | 'cardiovascular';
  timestamp: string;
  heartRate: number | null | undefined;
  readings?: WearableReading[];
  received?: boolean;
  onRegion?: (region: string) => void;
  onEmphasis?: (region: string | null) => void;
  heartRateDetail?: ReactNode;
  notification?: {alertId: string; acknowledged: boolean; referenceTime: string};
  provenanceNote?: string;
}) {
  const n = (v: number, d = 0) => v.toLocaleString('en-US', {maximumFractionDigits: d});
  const time = new Date(timestamp);
  const timeLabel = Number.isNaN(time.getTime()) ? 'Time unavailable' :
    time.toLocaleTimeString('en-GB', {hour: '2-digit', minute: '2-digit', timeZone: 'UTC'});
  const metrics: WearableReading[] = [
    {label: 'Heart rate', value: heartRate == null ? 'Not available' : n(heartRate),
      unit: heartRate == null ? undefined : 'bpm', region: mode === 'cardiovascular' ? 'Rhythm' : undefined,
      detail: heartRateDetail, testId: 'wearable-hr'},
    ...readings,
  ];
  const metricContent = (item: WearableReading) => {
    const Icon = item.label === 'Heart rate' ? Heart : item.label === 'Sleep' ? Moon :
      item.label === 'Steps' ? Footprints : Activity;
    return <>
      <span className="tc-metric-label"><Icon size={14} aria-hidden="true"/>{item.label}</span>
      <strong className="tc-metric-value" data-testid={item.testId}>
        {typeof item.value === 'number' ? n(item.value, 1) : item.value}
        {item.unit && <small>{item.unit}</small>}
      </strong>
      {item.detail && <span className="tc-metric-detail">{item.detail}</span>}
    </>;
  };
  return <section className="tc-wearable-card" aria-label="Live wearable" data-twin={mode}>
    <header className="tc-wearable-heading"><h2>Wearable</h2></header>
    <div className="tc-wearable-body">
      <div className={`tc-context-watch ${received ? 'tc-received' : ''}`}
        data-alert-id={notification?.alertId}
        aria-label={notification ? `Smartwatch: glucose risk alert${notification.acknowledged ? ', acknowledged' : ', delivered'}` : `Smartwatch: ${heartRate == null ? 'no heart-rate reading' : `${n(heartRate)} bpm`}`}
        style={{'--heartbeat': `${heartRate && heartRate > 0 ? 60 / heartRate : 1}s`} as CSSProperties}>
        <div className="tc-context-watch-strap" aria-hidden="true"/>
        <div className="tc-context-watch-shell"><div className="tc-context-watch-screen">
          <time dateTime={Number.isNaN(time.getTime()) ? undefined : timestamp} title="Reading time (UTC)">{timeLabel}</time>
          {notification ? <>
            {notification.acknowledged ? <Check size={20} aria-hidden="true"/> : <Bell size={20} aria-hidden="true"/>}
            <span className="tc-context-watch-alert">{notification.acknowledged ? 'Acknowledged' : 'Glucose risk'}</span>
          </> : <>
            <Heart size={16} fill="currentColor" className="tc-heart-pulse" aria-hidden="true"/>
            <div className="tc-context-watch-reading"><strong>{heartRate == null ? '—' : n(heartRate)}</strong><small>bpm</small></div>
          </>}
        </div></div>
        <div className="tc-context-watch-crown" aria-hidden="true"/>
      </div>
      <div className="tc-wearable-readings">
        {metrics.map(item => item.region && onRegion ?
          <button type="button" className="tc-wearable-metric" key={item.label}
            onClick={() => onRegion(item.region!)}
            onMouseEnter={() => onEmphasis?.(item.region!)} onMouseLeave={() => onEmphasis?.(null)}
            onFocus={() => onEmphasis?.(item.region!)} onBlur={() => onEmphasis?.(null)}>
            {metricContent(item)}
          </button> : <div className="tc-wearable-metric" key={item.label}>{metricContent(item)}</div>)}
      </div>
    </div>
    <small className="tc-wearable-provenance"><i aria-hidden="true"/>Wearable readings</small>
  </section>;
}

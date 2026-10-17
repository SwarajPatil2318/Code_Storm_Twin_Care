import { useLayoutEffect, useRef, useState } from "react";
import type { CSSProperties } from "react";
import { Activity, Droplets, HeartPulse, Leaf, RotateCcw } from "lucide-react";
import type { TwinState } from "./types";
export const HEART_ASSET = "/assets/anatomical-heart.png";
export function HeartVisual({
  state,
  onRegion,
  emphasizedRegion,
}: {
  state: TwinState;
  onRegion: (name: string) => void;
  emphasizedRegion?: string|null;
}) {
  const [layer, setLayer] = useState("Both");
  const stage = useRef<HTMLDivElement>(null);
  const [annotations, setAnnotations] = useState<{
    width: number;
    height: number;
    points: { startX: number; startY: number; x: number; y: number }[];
  } | null>(null);
  useLayoutEffect(() => {
    const element = stage.current;
    if (!element) return;
    const image = element.querySelector<HTMLElement>(".heart-image-wrap")!;
    const regions = [
      ["rhythm", 0.4, 0.24, "left"],
      ["fluid", 0.65, 0.4, "right"],
      ["recovery", 0.38, 0.69, "left"],
      ["activity", 0.68, 0.8, "right"],
    ] as const;
    const measure = () => {
      setAnnotations({
        width: element.clientWidth,
        height: element.clientHeight,
        points: regions.map(([name, x, y, side]) => {
          const button = element.querySelector<HTMLElement>(`.${name}`)!;
          return {
            startX:
              button.offsetLeft + (side === "left" ? button.offsetWidth : 0),
            startY: button.offsetTop + button.offsetHeight / 2,
            x: image.offsetLeft + image.offsetWidth * x,
            y: image.offsetTop + image.offsetHeight * y,
          };
        }),
      });
    };
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    observer.observe(image);
    regions.forEach(([name]) =>
      observer.observe(element.querySelector(`.${name}`)!),
    );
    measure();
    return () => observer.disconnect();
  }, []);
  const divergence = state.divergence.descriptive_tdi;
  const shift = divergence < 0.35 ? 2 : Math.min(24, (divergence - 0.25) * 36);
  const duration = 60 / (state.actual_state.heart_rate ?? 65);
  return (
    <div className="heart-stage" ref={stage} data-emphasis={emphasizedRegion??undefined}>
      <div className="heart-orbit" />
      <div className="heart-halo" />
      <div
        className="heart-image-wrap"
        style={
          { "--beat": `${duration}s`, "--shift": `${shift}px` } as CSSProperties
        }
      >
        {layer !== "Actual" && (
          <img
            className={`heart-image shadow-heart ${layer === "Shadow" ? "shadow-only" : ""}`}
            src={HEART_ASSET}
            alt={
              layer === "Shadow"
                ? "Shadow heart: symbolic expected physiology"
                : ""
            }
          />
        )}
        {layer !== "Shadow" && (
          <img
            className="heart-image actual-heart"
            src={HEART_ASSET}
            alt="Anatomical heart illustration representing observed physiology"
          />
        )}
      </div>
      {annotations && (
        <svg
          className="hotspot-lines"
          viewBox={`0 0 ${annotations.width} ${annotations.height}`}
          aria-hidden="true"
        >
          <path
            d={annotations.points
              .map((p) => `M${p.startX} ${p.startY} L${p.x} ${p.y}`)
              .join(" ")}
          />
          {annotations.points.map((p, i) => (
            <circle key={i} cx={p.x} cy={p.y} r="5.5" />
          ))}
        </svg>
      )}
      <button className="hotspot rhythm" onClick={() => onRegion("Rhythm")}>
        <HeartPulse size={19} />
        Rhythm
      </button>
      <button
        className="hotspot fluid"
        onClick={() => onRegion("Fluid & pressure")}
      >
        <Droplets size={19} />
        Fluid & pressure
      </button>
      <button className="hotspot recovery" onClick={() => onRegion("Recovery")}>
        <Leaf size={19} />
        Recovery
      </button>
      <button className="hotspot activity" onClick={() => onRegion("Activity")}>
        <Activity size={19} />
        Activity
      </button>
      <div className="heart-comparison-labels">
        <span className={layer === "Actual" ? "inactive" : ""}>
          <i className="cyan" />
          Expected twin
        </span>
        <span className={layer === "Shadow" ? "inactive" : ""}>
          <i className="coral" />
          Observed today
        </span>
      </div>
      <div className="heart-tools">
        <div className="segmented" aria-label="Heart comparison layers">
          {["Both", "Actual", "Shadow"].map((name) => (
            <button
              key={name}
              aria-pressed={layer === name}
              onClick={() => setLayer(name)}
            >
              {name === "Shadow" ? "Expected" : name}
            </button>
          ))}
        </div>
        <button
          className="icon-button"
          aria-label="Reset heart comparison"
          onClick={() => setLayer("Both")}
        >
          <RotateCcw size={15} />
        </button>
      </div>
    </div>
  );
}

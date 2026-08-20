import type { PredictionResult } from "../types";

export default function VerdictBanner({ result }: { result: PredictionResult }) {
  const healthy = result.is_healthy;
  return (
    <div className={`verdict ${healthy ? "healthy" : "damaged"}`}>
      <div className="verdict-icon">{healthy ? "\u2705" : "\u26A0\uFE0F"}</div>
      <div className="verdict-label">{result.verdict}</div>
      <div className="verdict-confidence">
        {result.confidence !== null
          ? `${(result.confidence * 100).toFixed(1)}% confidence`
          : "confidence unavailable"}
      </div>
    </div>
  );
}

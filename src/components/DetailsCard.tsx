import type { PredictionResult } from "../types";

export default function DetailsCard({ result }: { result: PredictionResult }) {
  const probs = result.probabilities
    ? Object.entries(result.probabilities)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 3)
    : [];

  return (
    <div className="details">
      <h2>Prediction: {result.label_name}</h2>

      <div className="detail-row">
        <span className="detail-label">Condition</span>
        <span className="detail-value">{result.label_name}</span>
      </div>
      <div className="detail-row">
        <span className="detail-label">Confidence</span>
        <span className="detail-value">
          {result.confidence !== null ? (result.confidence * 100).toFixed(2) + "%" : "N/A"}
        </span>
      </div>

      {probs.length > 0 && (
        <div className="prob-section">
          <h3>
            {probs.length <= 3 ? "Class probabilities" : "Top 3 probabilities"}
          </h3>
          {probs.map(([name, p]) => {
            const isHealthy = name.toLowerCase().includes("healthy");
            return (
              <div key={name} className="prob-row">
                <div className="prob-row-top">
                  <span className="prob-name">{name}</span>
                  <span className="prob-value">{(p * 100).toFixed(2)}%</span>
                </div>
                <div className="prob-bar-track">
                  <div
                    className={`prob-bar-fill ${isHealthy ? "healthy" : "damaged"}`}
                    style={{ width: `${Math.max(p * 100, 2)}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

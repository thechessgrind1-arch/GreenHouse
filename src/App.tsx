import { useState, useRef, useCallback } from "react";
import type { PredictionResult } from "./types";
import { CROPS } from "./types";
import VerdictBanner from "./components/VerdictBanner";
import DetailsCard from "./components/DetailsCard";

export default function App() {
  const [crop, setCrop] = useState(0);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((f: File) => {
    setFile(f);
    setResult(null);
    setError(null);
    setPreviewUrl(URL.createObjectURL(f));
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const f = e.dataTransfer.files[0];
      if (f && f.type.startsWith("image/")) handleFile(f);
    },
    [handleFile]
  );

  const predict = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("image", file);
      formData.append("crop", CROPS[crop].name);

      const res = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const body = await res.json();
        throw new Error(body.error || "Prediction failed");
      }
      const data: PredictionResult = await res.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-icon">
          <svg viewBox="0 0 24 24" fill="none" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
            <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19.2 2.96c1.4 9.3-2.8 15.04-8.2 17.04Z" />
            <path d="M2 21c0-3 1.85-5.36 5.08-6" />
          </svg>
        </div>
        <h1>GreenHouse Leaf Checker</h1>
        <p>Upload a leaf photo to check if it's healthy or damaged</p>
      </header>

      {/* Crop selector */}
      <div className="crop-selector">
        {CROPS.map((c, i) => (
          <button
            key={c.name}
            className={`crop-card ${crop === i ? "active" : ""}`}
            onClick={() => setCrop(i)}
          >
            <div className={`crop-card-icon ${c.icon}`}>
              {c.icon === "pepper" ? "\u{1F336}" : "\u{1F345}"}
            </div>
            <div>
              <div className="crop-card-name">{c.name}</div>
              <div className="crop-card-species">{c.species}</div>
            </div>
          </button>
        ))}
      </div>

      {/* Upload zone */}
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
      />
      <div
        className={`upload-zone ${dragging ? "dragging" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        {previewUrl ? (
          <>
            <img src={previewUrl} alt="Preview" className="upload-preview" />
            <div style={{ marginTop: "1rem", display: "flex", gap: "0.75rem" }}>
              <button className="btn btn-secondary" onClick={(e) => { e.stopPropagation(); reset(); }}>
                Remove
              </button>
            </div>
          </>
        ) : (
          <>
            <div className="upload-zone-icon">{"\u{1F4C2}"}</div>
            <div className="upload-zone-text">Drop a leaf image here or click to browse</div>
            <div className="upload-zone-hint">JPG, PNG, BMP, TIFF, WEBP</div>
          </>
        )}
      </div>

      {/* Predict button */}
      {file && !result && (
        <button className="btn btn-primary" onClick={predict} disabled={loading}>
          {loading ? (
            <>
              <span className="spinner" />
              Classifying...
            </>
          ) : (
            "Check Leaf"
          )}
        </button>
      )}

      {/* Error */}
      {error && <div className="error-box">{error}</div>}

      {/* Results */}
      {result && (
        <>
          <VerdictBanner result={result} />
          <DetailsCard result={result} />
          <button className="btn btn-secondary" onClick={reset}>
            Check Another Leaf
          </button>
        </>
      )}

      <footer className="footer">
        Screening aid only, not a substitute for agronomic diagnosis.
        <br />
        Each crop uses its own model — select the right one above.
      </footer>
    </div>
  );
}

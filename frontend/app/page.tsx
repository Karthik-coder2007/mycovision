"use client";

import { ChangeEvent, useMemo, useState } from "react";

type Result = {
  predicted_class: string;
  confidence: number;
  probabilities: Record<string, number>;
  gradcam_png: string;
  spatial_mask_png: string;
  note: string;
};

const API =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const sorted = useMemo(
    () =>
      result
        ? Object.entries(result.probabilities).sort(
            (a, b) => b[1] - a[1]
          )
        : [],
    [result]
  );

  function choose(e: ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];

    setResult(null);
    setError("");

    if (!f) return;

    if (
      !["image/jpeg", "image/png", "image/webp"].includes(f.type)
    ) {
      setError("Choose a JPG, PNG or WEBP image.");
      return;
    }

    setFile(f);
    setPreview(URL.createObjectURL(f));
  }

  async function predict() {
    if (!file) {
      setError("Upload a mushroom image first.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const fd = new FormData();
      fd.append("file", file);

      const r = await fetch(`${API}/predict`, {
        method: "POST",
        body: fd,
      });

      const d = await r.json();

      if (!r.ok) {
        throw new Error(d.detail || "Prediction failed");
      }

      setResult(d);
    } catch (e) {
      setError(
        e instanceof Error
          ? e.message
          : "Could not connect to model API."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="shell">
      <nav>
        <a className="brand" href="#">
          <span>M</span> MycoVision
        </a>
        <small>AI RESEARCH DEMO</small>
      </nav>

      <header>
        <div className="eyebrow">
          COMPUTER VISION · MUSHROOM CLASSIFICATION
        </div>

        <h1>
          Understand the <em>fungi</em>
          <br />
          in your image.
        </h1>

        <p>
          Upload a mushroom photo to view the predicted class,
          confidence, Grad-CAM overlay and learned spatial mask.
        </p>
      </header>

      <section className="grid">
        {/* Upload panel */}
        <article className="panel">
          <div className="panelhead">
            <h2>
              <i>01</i> Upload image
            </h2>
            <small>JPG · PNG · WEBP</small>
          </div>

          <label className="drop">
            {preview ? (
              <img src={preview} alt="Selected mushroom" />
            ) : (
              <div className="prompt">
                <b>＋</b>
                <strong>Choose a mushroom photo</strong>
                <span>Browse files from your device</span>
              </div>
            )}

            <input
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={choose}
            />
          </label>

          {file && (
            <p className="filename">
              {file.name} · {(file.size / 1048576).toFixed(2)} MB
            </p>
          )}

          <button onClick={predict} disabled={!file || loading}>
            {loading ? "Analyzing image…" : "Analyze mushroom ↗"}
          </button>

          {error && <div className="error">{error}</div>}

          <p className="hint">
            The image is sent to the configured Python model API.
          </p>
        </article>

        {/* Results panel */}
        <article className="panel">
          <div className="panelhead">
            <h2>
              <i>02</i> Analysis results
            </h2>
            <small>{result ? "COMPLETE" : "AWAITING IMAGE"}</small>
          </div>

          {!result ? (
            <div className="empty">
              <b>✳</b>
              <strong>Your results will appear here</strong>
              <span>Upload an image and run the model.</span>
            </div>
          ) : (
            <div className="results">
              <div className="prediction">
                <div>
                  <small>PREDICTED CLASS</small>
                  <h3>{result.predicted_class}</h3>
                </div>

                <strong>
                  {(result.confidence * 100).toFixed(1)}
                  <small>%</small>
                </strong>
              </div>

              <div className="track">
                <span
                  style={{
                    width: `${result.confidence * 100}%`,
                  }}
                />
              </div>

              {/* Visualization overlays */}
              <div className="visuals">
                <div>
                  <h4>Grad-CAM Overlay</h4>

                  <img
                    src={result.gradcam_png}
                    alt="Grad-CAM heatmap blended with the original mushroom"
                  />

                  <p>
                    Heatmap blended with the original image to highlight
                    regions associated with the prediction.
                  </p>
                </div>

                <div>
                  <h4>Spatial Mask Overlay</h4>

                  <img
                    src={result.spatial_mask_png}
                    alt="Spatial attention mask blended with the original mushroom"
                  />

                  <p>
                    Learned spatial mask blended with the original image.
                  </p>
                </div>
              </div>

              <h4 className="probtitle">CLASS PROBABILITIES</h4>

              {sorted.map(([name, v]) => (
                <div className="prob" key={name}>
                  <span>{name}</span>

                  <div className="mini">
                    <span
                      style={{
                        width: `${v * 100}%`,
                      }}
                    />
                  </div>

                  <b>{(v * 100).toFixed(1)}%</b>
                </div>
              ))}

              <p className="warning">
                {result.note} Not a food-safety tool. Never eat
                mushrooms based on an AI prediction.
              </p>
            </div>
          )}
        </article>
      </section>

      <footer>
        MYCOVISION / COMPUTER VISION PROJECT
        <span>AI predictions can be wrong</span>
      </footer>
    </main>
  );
}
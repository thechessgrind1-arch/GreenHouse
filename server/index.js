import http from "node:http";
import { spawn } from "node:child_process";
import { writeFile, mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import os from "node:os";
import crypto from "node:crypto";

const PORT = 3001;
const PYTHON = process.env.PYTHON || "python3";
const PROJECT_ROOT = path.resolve(import.meta.dirname, "..");

const TOMATO_CLASS_NAMES = [
  "Tomato___Bacterial_spot",
  "Tomato___Early_blight",
  "Tomato___Late_blight",
  "Tomato___Leaf_Mold",
  "Tomato___Septoria_leaf_spot",
  "Tomato___Spider_mites Two-spotted_spider_mite",
  "Tomato___Target_Spot",
  "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
  "Tomato___Tomato_mosaic_virus",
  "Tomato___healthy",
];

function prettify(rawName) {
  let name = rawName.replace("Tomato___", "").replace("Tomato_", "");
  name = name.replace(/_/g, " ").trim();
  return name
    .split(" ")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

function parseMultipart(buf, boundary) {
  const parts = buf.split(Buffer.from(`--${boundary}`));
  const fields = {};
  let imageBuf = null;
  let imageExt = "jpg";

  for (const part of parts) {
    if (part.length < 10 || part.toString().trim() === "--") continue;
    const headerEnd = part.indexOf(Buffer.from("\r\n\r\n"));
    if (headerEnd === -1) continue;
    const headerStr = part.slice(0, headerEnd).toString();
    const body = part.slice(headerEnd + 4, part.length - 2);

    const nameMatch = headerStr.match(/name="([^"]+)"/);
    if (!nameMatch) continue;
    const fieldName = nameMatch[1];

    if (headerStr.includes("filename=")) {
      const extMatch = headerStr.match(/filename="[^"]+\.(\w+)"/);
      if (extMatch) imageExt = extMatch[1].toLowerCase();
      imageBuf = body;
    } else {
      fields[fieldName] = body.toString().trim();
    }
  }
  return { fields, imageBuf, imageExt };
}

function runPython(args) {
  return new Promise((resolve, reject) => {
    const proc = spawn(PYTHON, args, {
      cwd: PROJECT_ROOT,
      env: { ...process.env, KERAS_BACKEND: "jax" },
    });
    let stdout = "";
    let stderr = "";
    proc.stdout.on("data", (d) => (stdout += d));
    proc.stderr.on("data", (d) => (stderr += d));
    proc.on("close", (code) => {
      if (code !== 0) reject(new Error(stderr || stdout || `exit ${code}`));
      else resolve(stdout);
    });
    proc.on("error", reject);
  });
}

async function predictPepper(imagePath) {
  const script = `
import json, sys
sys.path.insert(0, ".")
from predict_pepper import predict
result = predict(${JSON.stringify(imagePath)})
print("___RESULT___" + json.dumps(result))
`;
  const out = await runPython(["-c", script]);
  const marker = out.indexOf("___RESULT___");
  if (marker === -1) throw new Error("No result from pepper predictor");
  return JSON.parse(out.slice(marker + 13).trim());
}

async function predictTomato(imagePath) {
  const script = `
import json, os, sys
os.environ.setdefault("KERAS_BACKEND", "jax")
sys.path.insert(0, ".")
import numpy as np
from PIL import Image

model_path = "app/tomato_model.keras"
try:
    import keras
except ImportError:
    from tensorflow import keras

model = keras.models.load_model(model_path)
img = Image.fromarray(__import__("cv2").imread(${JSON.stringify(imagePath)}))
img = img.convert("RGB").resize((224, 224), Image.BILINEAR)
batch = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
probs = model.predict(batch, verbose=0)[0]
idx = int(np.argmax(probs))
names = ${JSON.stringify(TOMATO_CLASS_NAMES)}
raw = names[idx]
healthy = raw.endswith("healthy")
result = {
    "verdict": "HEALTHY" if healthy else "DAMAGED",
    "is_healthy": healthy,
    "label": idx,
    "label_name": __import__("app", fromlist=["prettify"]).prettify(raw) if False else "${prettify("PLACEHOLDER")}",
    "confidence": float(probs[idx]),
    "probabilities": {__import__("app", fromlist=["prettify"]).prettify(n) if False else n.replace("Tomato___","").replace("_"," ").title(): float(p) for n, p in zip(names, probs)},
}
print("___RESULT___" + json.dumps(result))
`;
  // Simplified: just build the result inline
  const script2 = `
import json, os, sys
os.environ.setdefault("KERAS_BACKEND", "jax")
sys.path.insert(0, ".")
import numpy as np
import cv2
from PIL import Image

model_path = "app/tomato_model.keras"
try:
    import keras
except ImportError:
    from tensorflow import keras

model = keras.models.load_model(model_path)
bgr = cv2.imread(${JSON.stringify(imagePath)})
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
img = Image.fromarray(rgb).convert("RGB").resize((224, 224), Image.BILINEAR)
batch = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
probs = model.predict(batch, verbose=0)[0]
idx = int(np.argmax(probs))
names = ${JSON.stringify(TOMATO_CLASS_NAMES)}

def prettify(raw):
    n = raw.replace("Tomato___", "").replace("Tomato_", "")
    n = n.replace("_", " ").strip()
    return " ".join(w.capitalize() for w in n.split())

raw = names[idx]
healthy = raw.endswith("healthy")
result = {
    "verdict": "HEALTHY" if healthy else "DAMAGED",
    "is_healthy": healthy,
    "label": idx,
    "label_name": prettify(raw),
    "confidence": float(probs[idx]),
    "probabilities": {prettify(n): float(p) for n, p in zip(names, probs)},
}
print("___RESULT___" + json.dumps(result))
`;
  const out = await runPython(["-c", script2]);
  const marker = out.indexOf("___RESULT___");
  if (marker === -1) throw new Error("No result from tomato predictor");
  return JSON.parse(out.slice(marker + 13).trim());
}

const server = http.createServer(async (req, res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    res.writeHead(204);
    res.end();
    return;
  }

  if (req.method === "POST" && req.url === "/api/predict") {
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    const buf = Buffer.concat(chunks);

    const boundaryMatch = (req.headers["content-type"] || "")
      .match(/boundary=(.+)/);
    if (!boundaryMatch) {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Missing multipart boundary" }));
      return;
    }

    const { fields, imageBuf, imageExt } = parseMultipart(
      buf,
      boundaryMatch[1].trim()
    );
    const crop = fields.crop || "Bell pepper";

    if (!imageBuf) {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "No image provided" }));
      return;
    }

    const tmpDir = path.join(os.tmpdir(), "greenhouse-uploads");
    if (!existsSync(tmpDir)) await mkdir(tmpDir, { recursive: true });
    const imageName = `upload_${crypto.randomUUID()}.${imageExt}`;
    const imagePath = path.join(tmpDir, imageName);
    await writeFile(imagePath, imageBuf);

    try {
      let result;
      if (crop === "Tomato") {
        result = await predictTomato(imagePath);
      } else {
        result = await predictPepper(imagePath);
      }
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify(result));
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Prediction failed";
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: msg }));
    }
    return;
  }

  res.writeHead(404, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ error: "Not found" }));
});

server.listen(PORT, () => {
  console.log(`GreenHouse API server on http://localhost:${PORT}`);
});

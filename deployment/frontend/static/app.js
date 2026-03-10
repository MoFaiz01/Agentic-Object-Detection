const fileInput = document.getElementById("fileInput");
const detectBtn = document.getElementById("detectBtn");
const clearBtn = document.getElementById("clearBtn");
const statusText = document.getElementById("statusText");
const detCount = document.getElementById("detCount");
const latencyText = document.getElementById("latency");
const detectionList = document.getElementById("detectionList");
const rawResponse = document.getElementById("rawResponse");
const canvas = document.getElementById("previewCanvas");
const ctx = canvas.getContext("2d");

let selectedFile = null;
let imageBitmap = null;
let currentDetections = [];

function setStatus(message, isError = false) {
  statusText.textContent = message;
  statusText.classList.toggle("error", isError);
}

async function loadFilePreview(file) {
  const objectUrl = URL.createObjectURL(file);
  imageBitmap = await createImageBitmap(file);
  canvas.width = imageBitmap.width;
  canvas.height = imageBitmap.height;
  drawScene();
  URL.revokeObjectURL(objectUrl);
}

function drawScene() {
  if (!imageBitmap) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(imageBitmap, 0, 0);

  currentDetections.forEach((det, idx) => {
    const bbox = det.bbox || [0, 0, 0, 0];
    const [x1, y1, x2, y2] = bbox;
    const width = x2 - x1;
    const height = y2 - y1;

    ctx.lineWidth = 3;
    ctx.strokeStyle = "#ffb703";
    ctx.strokeRect(x1, y1, width, height);

    const label = `${det.class_name || "object"} ${(det.score ?? 0).toFixed(2)} #${idx + 1}`;
    ctx.font = "16px Segoe UI";
    const textWidth = ctx.measureText(label).width + 12;
    const labelY = Math.max(y1 - 28, 4);

    ctx.fillStyle = "rgba(0, 95, 115, 0.95)";
    ctx.fillRect(x1, labelY, textWidth, 24);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, x1 + 6, labelY + 17);
  });
}

function renderDetections(detections) {
  detectionList.innerHTML = "";
  if (!detections.length) {
    const li = document.createElement("li");
    li.textContent = "No detections.";
    detectionList.appendChild(li);
    return;
  }

  detections.forEach((det, idx) => {
    const li = document.createElement("li");
    const bbox = det.bbox || [0, 0, 0, 0];
    li.textContent =
      `#${idx + 1} ${det.class_name || "object"} | score=${(det.score ?? 0).toFixed(3)} | bbox=[${bbox.join(", ")}]`;
    detectionList.appendChild(li);
  });
}

async function runDetection() {
  if (!selectedFile) {
    setStatus("Select an image first.", true);
    return;
  }

  detectBtn.disabled = true;
  setStatus("Running detection...");

  const formData = new FormData();
  formData.append("file", selectedFile);

  const startedAt = performance.now();
  try {
    const response = await fetch("/detect", {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Request failed: ${response.status} ${errorText}`);
    }

    const payload = await response.json();
    const detections = payload.detections || [];
    const backend = payload.backend || "unknown";
    const weights = payload.weights || "unknown";
    const elapsedMs = Math.round(performance.now() - startedAt);

    currentDetections = detections;
    drawScene();
    renderDetections(detections);

    detCount.textContent = String(detections.length);
    latencyText.textContent = `${elapsedMs} ms`;
    rawResponse.textContent = JSON.stringify(payload, null, 2);
    setStatus(`Done. ${detections.length} detection(s) found. backend=${backend}, weights=${weights}`);
  } catch (error) {
    setStatus(error.message || "Detection failed.", true);
  } finally {
    detectBtn.disabled = false;
  }
}

function clearUI() {
  selectedFile = null;
  imageBitmap = null;
  currentDetections = [];
  fileInput.value = "";
  detCount.textContent = "0";
  latencyText.textContent = "-";
  detectionList.innerHTML = "";
  rawResponse.textContent = "";
  canvas.width = 0;
  canvas.height = 0;
  drawScene();
  setStatus("Waiting for image...");
  detectBtn.disabled = true;
}

fileInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) {
    clearUI();
    return;
  }

  selectedFile = file;
  currentDetections = [];
  detCount.textContent = "0";
  latencyText.textContent = "-";
  detectionList.innerHTML = "";
  rawResponse.textContent = "";

  try {
    await loadFilePreview(file);
    detectBtn.disabled = false;
    setStatus("Image loaded. Click Run Detection.");
  } catch (error) {
    clearUI();
    setStatus("Could not load image preview.", true);
  }
});

detectBtn.addEventListener("click", runDetection);
clearBtn.addEventListener("click", clearUI);

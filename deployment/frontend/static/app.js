const fileInput = document.getElementById("fileInput");
const detectBtn = document.getElementById("detectBtn");
const startLiveBtn = document.getElementById("startLiveBtn");
const stopLiveBtn = document.getElementById("stopLiveBtn");
const clearBtn = document.getElementById("clearBtn");
const statusText = document.getElementById("statusText");
const detCount = document.getElementById("detCount");
const latencyText = document.getElementById("latency");
const modeText = document.getElementById("modeText");
const detectionList = document.getElementById("detectionList");
const rawResponse = document.getElementById("rawResponse");
const canvas = document.getElementById("previewCanvas");
const ctx = canvas.getContext("2d");
const liveVideo = document.getElementById("liveVideo");

let selectedFile = null;
let imageBitmap = null;
let currentDetections = [];
let liveStream = null;
let liveMode = false;
let liveLoopHandle = null;
let liveRequestInFlight = false;

const LIVE_INTERVAL_MS = 220;

function setStatus(message, isError = false) {
  statusText.textContent = message;
  statusText.classList.toggle("error", isError);
}

function setMode(mode) {
  modeText.textContent = mode;
}

function drawDetections() {
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

async function loadFilePreview(file) {
  imageBitmap = await createImageBitmap(file);
  canvas.width = imageBitmap.width;
  canvas.height = imageBitmap.height;
  drawScene();
}

function drawScene() {
  if (!imageBitmap) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(imageBitmap, 0, 0);
  drawDetections();
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

async function postBlobForDetection(blob) {
  const formData = new FormData();
  formData.append("file", blob, "frame.jpg");

  const startedAt = performance.now();
  const response = await fetch("/detect", {
    method: "POST",
    body: formData
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Request failed: ${response.status} ${errorText}`);
  }

  const payload = await response.json();
  const elapsedMs = Math.round(performance.now() - startedAt);
  const detections = payload.detections || [];

  currentDetections = detections;
  detCount.textContent = String(detections.length);
  latencyText.textContent = `${elapsedMs} ms`;
  renderDetections(detections);
  rawResponse.textContent = JSON.stringify(payload, null, 2);

  return { detections, payload };
}

async function runDetection() {
  if (!selectedFile) {
    setStatus("Select an image first.", true);
    return;
  }

  detectBtn.disabled = true;
  setStatus("Running detection...");

  try {
    const result = await postBlobForDetection(selectedFile);
    drawScene();
    setStatus(`Done. ${result.detections.length} detection(s) found.`);
  } catch (error) {
    setStatus(error.message || "Detection failed.", true);
  } finally {
    detectBtn.disabled = false;
  }
}

function frameToBlob() {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) {
        reject(new Error("Could not capture camera frame."));
        return;
      }
      resolve(blob);
    }, "image/jpeg", 0.85);
  });
}

async function liveTick() {
  if (!liveMode || !liveVideo.videoWidth || !liveVideo.videoHeight) {
    return;
  }

  if (liveRequestInFlight) {
    return;
  }

  liveRequestInFlight = true;
  try {
    canvas.width = liveVideo.videoWidth;
    canvas.height = liveVideo.videoHeight;
    ctx.drawImage(liveVideo, 0, 0, canvas.width, canvas.height);

    const blob = await frameToBlob();
    const result = await postBlobForDetection(blob);

    ctx.drawImage(liveVideo, 0, 0, canvas.width, canvas.height);
    drawDetections();
    setStatus(`Live: ${result.detections.length} detection(s)`);
  } catch (error) {
    setStatus(error.message || "Live detection failed.", true);
    await stopLiveCamera(false);
  } finally {
    liveRequestInFlight = false;
  }
}

async function startLiveCamera() {
  if (liveMode) {
    return;
  }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus("Browser does not support camera access.", true);
    return;
  }

  try {
    try {
      liveStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: false
      });
    } catch (_) {
      liveStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    }
  } catch (error) {
    setStatus(error.message || "Could not access camera.", true);
    return;
  }

  liveVideo.srcObject = liveStream;
  await liveVideo.play();

  liveMode = true;
  selectedFile = null;
  imageBitmap = null;
  currentDetections = [];
  fileInput.value = "";

  detectBtn.disabled = true;
  startLiveBtn.disabled = true;
  stopLiveBtn.disabled = false;
  fileInput.disabled = true;
  liveVideo.classList.add("active");
  setMode("Live Camera");
  setStatus("Live camera started. Running detections...");

  if (liveLoopHandle) {
    clearInterval(liveLoopHandle);
  }
  liveLoopHandle = setInterval(liveTick, LIVE_INTERVAL_MS);
  await liveTick();
}

async function stopLiveCamera(showStatus = true) {
  liveMode = false;
  if (liveLoopHandle) {
    clearInterval(liveLoopHandle);
    liveLoopHandle = null;
  }

  if (liveStream) {
    liveStream.getTracks().forEach((track) => track.stop());
    liveStream = null;
  }

  liveVideo.pause();
  liveVideo.srcObject = null;
  liveVideo.classList.remove("active");
  startLiveBtn.disabled = false;
  stopLiveBtn.disabled = true;
  fileInput.disabled = false;
  setMode("Idle");

  if (showStatus) {
    setStatus("Live camera stopped.");
  }
}

async function clearUI() {
  await stopLiveCamera(false);

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
  setMode("Idle");
  setStatus("Waiting for image or camera...");
  detectBtn.disabled = true;
}

fileInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) {
    await clearUI();
    return;
  }

  if (liveMode) {
    await stopLiveCamera();
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
    setMode("Image");
    setStatus("Image loaded. Click Run Detection.");
  } catch (error) {
    await clearUI();
    setStatus("Could not load image preview.", true);
  }
});

detectBtn.addEventListener("click", runDetection);
startLiveBtn.addEventListener("click", startLiveCamera);
stopLiveBtn.addEventListener("click", stopLiveCamera);
clearBtn.addEventListener("click", clearUI);

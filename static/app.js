const fileInput = document.getElementById("fileInput");
const dropZone = document.getElementById("dropZone");
const fileMeta = document.getElementById("fileMeta");
const inputWrap = document.getElementById("inputWrap");
const inputVideo = document.getElementById("inputVideo");
const convertBtn = document.getElementById("convertBtn");
const cancelBtn = document.getElementById("cancelBtn");
const statusLabel = document.getElementById("statusLabel");
const statusPct = document.getElementById("statusPct");
const statusMessage = document.getElementById("statusMessage");
const barFill = document.getElementById("barFill");
const sourcePlayer = document.getElementById("sourcePlayer");
const depthPlayer = document.getElementById("depthPlayer");
const downloadBtn = document.getElementById("downloadBtn");
const deviceChip = document.getElementById("deviceChip");

let selectedFile = null;
let currentJob = null;
let eventSource = null;

function setStatus(label, message, progress) {
  statusLabel.textContent = label;
  statusMessage.textContent = message;
  const pct = Math.max(0, Math.min(100, Math.round((progress || 0) * 100)));
  statusPct.textContent = `${pct}%`;
  barFill.style.width = `${pct}%`;
}

function prettySize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function useFile(file) {
  selectedFile = file;
  fileMeta.textContent = `${file.name} · ${prettySize(file.size)}`;
  const url = URL.createObjectURL(file);
  inputVideo.src = url;
  sourcePlayer.src = url;
  inputWrap.hidden = false;
  convertBtn.disabled = false;
  downloadBtn.hidden = true;
  depthPlayer.removeAttribute("src");
  setStatus("Ready", "File loaded. Choose a model and generate a depth video.", 0);
}

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragover");
  const file = event.dataTransfer.files[0];
  if (file) useFile(file);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) useFile(fileInput.files[0]);
});

async function loadHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    deviceChip.textContent = data.device_name || data.device;
    const banner = document.getElementById("torchBanner");
    if (banner) banner.hidden = Boolean(data.torch);
  } catch {
    deviceChip.textContent = "Server offline";
  }
}

function formPayload() {
  const body = new FormData();
  body.append("file", selectedFile);
  body.append("encoder", document.getElementById("encoder").value);
  body.append("metric", document.getElementById("metric").value);
  body.append("colormap", document.getElementById("colormap").value);
  body.append("layout", document.getElementById("layout").value);
  body.append("max_res", document.getElementById("max_res").value);
  body.append("target_fps", document.getElementById("target_fps").value);
  body.append("max_len", document.getElementById("max_len").value);
  body.append("input_size", document.getElementById("input_size").value);
  body.append("mode", document.getElementById("mode").value);
  body.append("keep_audio", document.getElementById("keep_audio").checked);
  body.append("invert", document.getElementById("invert").checked);
  body.append("use_fp16", document.getElementById("use_fp16").checked);
  body.append("grayscale", document.getElementById("colormap").value === "gray");
  return body;
}

function listen(jobId) {
  if (eventSource) eventSource.close();
  eventSource = new EventSource(`/api/jobs/${jobId}/events`);
  eventSource.onmessage = (event) => {
    const job = JSON.parse(event.data);
    applyJob(job);
    if (["done", "error", "cancelled"].includes(job.status)) {
      eventSource.close();
      eventSource = null;
    }
  };
}

function applyJob(job) {
  currentJob = job.id;
  const labels = {
    queued: "Queued",
    running: "Converting",
    done: "Done",
    error: "Failed",
    cancelled: "Cancelled",
  };
  setStatus(labels[job.status] || job.status, job.message, job.progress);
  cancelBtn.hidden = !["queued", "running"].includes(job.status);
  convertBtn.disabled = ["queued", "running"].includes(job.status);
  if (job.status === "done") {
    depthPlayer.src = `/api/jobs/${job.id}/video?t=${Date.now()}`;
    downloadBtn.href = `/api/jobs/${job.id}/download`;
    downloadBtn.hidden = false;
    convertBtn.disabled = !selectedFile;
  }
}

convertBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  convertBtn.disabled = true;
  setStatus("Uploading", "Sending video to the converter…", 0.02);
  try {
    const res = await fetch("/api/jobs", { method: "POST", body: formPayload() });
    if (!res.ok) throw new Error(await res.text());
    const job = await res.json();
    applyJob(job);
    listen(job.id);
  } catch (error) {
    setStatus("Failed", error.message || "Could not start conversion", 0);
    convertBtn.disabled = !selectedFile;
  }
});

cancelBtn.addEventListener("click", async () => {
  if (!currentJob) return;
  await fetch(`/api/jobs/${currentJob}/cancel`, { method: "POST" });
});

loadHealth();

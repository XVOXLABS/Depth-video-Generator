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
let lastJob = null;
let torchReady = false;
let ffmpegReady = false;

function toolsReady() {
  return torchReady && ffmpegReady;
}

function formatElapsed(seconds) {
  const total = Math.max(0, Math.round(seconds || 0));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function setStatus(label, message, progress) {
  const elapsed = lastJob && ["queued", "running"].includes(lastJob.status)
    ? ` · ${formatElapsed(lastJob.elapsed_s)}`
    : "";
  statusLabel.textContent = label + elapsed;
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
  convertBtn.disabled = !toolsReady();
  downloadBtn.hidden = true;
  depthPlayer.removeAttribute("src");
  if (!torchReady) {
    setStatus("Setup needed", "Install PyTorch with python3 -m pip install torch torchvision, then restart the app.", 0);
    return;
  }
  if (!ffmpegReady) {
    setStatus("Setup needed", "Install FFmpeg with brew install ffmpeg, then restart the app.", 0);
    return;
  }
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
    torchReady = Boolean(data.torch);
    ffmpegReady = data.ffmpeg !== false;
    deviceChip.textContent = data.device_name || data.device;
    const torchBanner = document.getElementById("torchBanner");
    if (torchBanner) torchBanner.hidden = torchReady;
    const ffmpegBanner = document.getElementById("ffmpegBanner");
    if (ffmpegBanner) ffmpegBanner.hidden = ffmpegReady;
    const cpuBanner = document.getElementById("cpuBanner");
    if (cpuBanner) cpuBanner.hidden = !data.cpu_inference || !torchReady;
    if (!toolsReady()) {
      convertBtn.disabled = true;
    } else if (selectedFile) {
      convertBtn.disabled = false;
    }
    await restoreLastJob();
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

let pollTimer = null;
let shownResultId = null;
const LAST_JOB_KEY = "depthVideoLastJobId";

function stopListening() {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

function showResult(job) {
  downloadBtn.href = `/api/jobs/${job.id}/download`;
  downloadBtn.hidden = false;
  downloadBtn.removeAttribute("hidden");
  const hint = document.getElementById("resultHint");
  if (hint) {
    hint.hidden = false;
    hint.textContent = "Finished. Preview is on the right — or click Download depth MP4 to save the file.";
  }
  if (shownResultId === job.id && depthPlayer.getAttribute("src")) {
    return;
  }
  shownResultId = job.id;
  const url = `/api/jobs/${job.id}/video?t=${Date.now()}`;
  depthPlayer.removeAttribute("src");
  depthPlayer.src = url;
  depthPlayer.load();
  const play = () => depthPlayer.play().catch(() => {});
  depthPlayer.addEventListener("loadeddata", play, { once: true });
  depthPlayer.onerror = () => {
    if (hint) {
      hint.hidden = false;
      hint.textContent = "The file is ready on disk, but this browser could not preview it. Click Download depth MP4.";
    }
  };
}

function applyJob(job) {
  lastJob = job;
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
  convertBtn.disabled = ["queued", "running"].includes(job.status) || !toolsReady() || !selectedFile;
  if (job.status === "done") {
    localStorage.setItem(LAST_JOB_KEY, job.id);
    showResult(job);
    convertBtn.disabled = !selectedFile || !toolsReady();
    stopListening();
  }
}

function listen(jobId) {
  stopListening();
  localStorage.setItem(LAST_JOB_KEY, jobId);
  eventSource = new EventSource(`/api/jobs/${jobId}/events`);
  eventSource.onmessage = (event) => {
    const job = JSON.parse(event.data);
    applyJob(job);
  };
  eventSource.onerror = () => {
    if (lastJob && ["done", "error", "cancelled"].includes(lastJob.status)) {
      stopListening();
    }
  };
  pollTimer = setInterval(async () => {
    try {
      const res = await fetch(`/api/jobs/${jobId}`);
      if (!res.ok) return;
      applyJob(await res.json());
    } catch {
      /* keep polling until the job finishes */
    }
  }, 1000);
}

async function restoreLastJob() {
  const jobId = localStorage.getItem(LAST_JOB_KEY);
  if (!jobId) return;
  try {
    const res = await fetch(`/api/jobs/${jobId}`);
    if (!res.ok) return;
    applyJob(await res.json());
  } catch {
    /* ignore */
  }
}

convertBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  if (!torchReady) {
    setStatus("Setup needed", "Install PyTorch first: python3 -m pip install torch torchvision", 0);
    return;
  }
  if (!ffmpegReady) {
    setStatus("Setup needed", "Install FFmpeg first: brew install ffmpeg", 0);
    return;
  }
  convertBtn.disabled = true;
  shownResultId = null;
  downloadBtn.hidden = true;
  depthPlayer.removeAttribute("src");
  setStatus("Uploading", "Sending video to the converter…", 0.02);
  try {
    const res = await fetch("/api/jobs", { method: "POST", body: formPayload() });
    if (!res.ok) throw new Error(await res.text());
    const job = await res.json();
    applyJob(job);
    listen(job.id);
  } catch (error) {
    setStatus("Failed", error.message || "Could not start conversion", 0);
    convertBtn.disabled = !selectedFile || !toolsReady();
  }
});

cancelBtn.addEventListener("click", async () => {
  if (!currentJob) return;
  await fetch(`/api/jobs/${currentJob}/cancel`, { method: "POST" });
});

loadHealth();

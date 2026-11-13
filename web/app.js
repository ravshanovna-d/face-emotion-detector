// Face detection (YOLOv8n-face) + emotion recognition (HSEmotion EfficientNet-B0) in the browser.
// Every pre/post-processing step mirrors face_pipeline.py, which train.py verifies against
// ultralytics + hsemotion_onnx.

const $ = (id) => document.getElementById(id);
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const EMO_SIZE = 224;
const PAD = 114;
const MAX_SIDE = 1600;        // large uploads are downscaled for speed
const MAX_FACES = 20;

let meta, detSession, emoSession;
let SIZE = 320, CONF = 0.25, IOU = 0.7, EMOTIONS = [];
let faces = [], selected = -1, current = null;   // current = {canvas, w, h}
let stream = null, running = false;

const canvas = $("canvas");
const ctx = canvas.getContext("2d");
const work = document.createElement("canvas");
const wctx = work.getContext("2d", { willReadFrequently: true });

// ---------- image maths (same as face_pipeline.py) ----------
function axis(nOut, nIn, offset, len) {
  // cv2.INTER_LINEAR: half-pixel centres, clamped at the edges. Indices are absolute (offset + local).
  const i0 = new Int32Array(nOut), i1 = new Int32Array(nOut), fr = new Float32Array(nOut);
  const scale = len / nOut;
  for (let o = 0; o < nOut; o++) {
    let f = (o + 0.5) * scale - 0.5;
    let a = Math.floor(f);
    let t = f - a;
    if (a < 0) { a = 0; t = 0; }
    if (a >= len - 1) { a = len - 1; t = 0; }
    i0[o] = offset + a; i1[o] = offset + Math.min(a + 1, len - 1); fr[o] = t;
  }
  return [i0, i1, fr];
}

// Bilinear resize of the region (x, y, w, h) of an RGBA buffer; calls put(index, r, g, b) with uint8 values.
function resizeRegion(src, srcW, x, y, w, h, outW, outH, put) {
  const [x0, x1, fx] = axis(outW, w, x, w);
  const [y0, y1, fy] = axis(outH, h, y, h);
  for (let oy = 0; oy < outH; oy++) {
    const r0 = y0[oy] * srcW, r1 = y1[oy] * srcW, ty = fy[oy];
    for (let ox = 0; ox < outW; ox++) {
      const tx = fx[ox];
      const a = (r0 + x0[ox]) * 4, b = (r0 + x1[ox]) * 4, c = (r1 + x0[ox]) * 4, d = (r1 + x1[ox]) * 4;
      const px = [0, 0, 0];
      for (let k = 0; k < 3; k++) {
        const top = src[a + k] * (1 - tx) + src[b + k] * tx;
        const bot = src[c + k] * (1 - tx) + src[d + k] * tx;
        px[k] = Math.min(255, Math.max(0, Math.floor(top * (1 - ty) + bot * ty + 0.5)));
      }
      put(oy * outW + ox, px[0], px[1], px[2]);
    }
  }
}

const pyRound = (v) => {               // Python round(): halves go to the even neighbour
  const r = Math.round(v);
  return Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
};

function letterbox(rgba, w, h) {
  const r = Math.min(SIZE / h, SIZE / w);
  const nw = Math.max(pyRound(w * r), 1), nh = Math.max(pyRound(h * r), 1);
  const left = pyRound((SIZE - nw) / 2 - 0.1), top = pyRound((SIZE - nh) / 2 - 0.1);
  const area = SIZE * SIZE;
  const data = new Float32Array(3 * area).fill(PAD / 255);
  resizeRegion(rgba, w, 0, 0, w, h, nw, nh, (i, R, G, B) => {
    const oy = Math.floor(i / nw), ox = i - oy * nw;
    const j = (oy + top) * SIZE + ox + left;
    data[j] = R / 255; data[area + j] = G / 255; data[2 * area + j] = B / 255;
  });
  return { tensor: new ort.Tensor("float32", data, [1, 3, SIZE, SIZE]), gx: nw / w, gy: nh / h, left, top };
}

function iou(a, b) {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  const area = (z) => (z[2] - z[0]) * (z[3] - z[1]);
  return inter / (area(a) + area(b) - inter + 1e-9);
}

function decode(out, lb, w, h) {
  const [, ch, n] = out.dims, p = out.data;
  let cands = [];
  for (let i = 0; i < n; i++) {
    const s = p[4 * n + i];
    if (s <= CONF) continue;
    const cx = p[i], cy = p[n + i], bw = p[2 * n + i], bh = p[3 * n + i];
    cands.push({ box: [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], score: s });
  }
  cands.sort((a, b) => b.score - a.score);
  const keep = [];
  while (cands.length) {
    const best = cands.shift();
    keep.push(best);
    cands = cands.filter((c) => iou(best.box, c.box) <= IOU);
  }
  const clip = (v, hi) => Math.min(Math.max(v, 0), hi);
  return keep.map((k) => {
    const b = k.box;
    return {
      box: [clip((b[0] - lb.left) / lb.gx, w), clip((b[1] - lb.top) / lb.gy, h),
            clip((b[2] - lb.left) / lb.gx, w), clip((b[3] - lb.top) / lb.gy, h)],
      score: k.score,
    };
  });
}

function emotionBatch(rgba, w, list) {
  const area = EMO_SIZE * EMO_SIZE, per = 3 * area;
  const data = new Float32Array(per * list.length);
  list.forEach((f, n) => {
    const [x1, y1, x2, y2] = f.box.map(Math.trunc);
    const base = n * per;
    resizeRegion(rgba, w, x1, y1, x2 - x1, y2 - y1, EMO_SIZE, EMO_SIZE, (i, R, G, B) => {
      data[base + i] = (R / 255 - MEAN[0]) / STD[0];
      data[base + area + i] = (G / 255 - MEAN[1]) / STD[1];
      data[base + 2 * area + i] = (B / 255 - MEAN[2]) / STD[2];
    });
  });
  return new ort.Tensor("float32", data, [list.length, 3, EMO_SIZE, EMO_SIZE]);
}

function softmax(z) {
  const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

// ---------- pipeline ----------
async function analyze(w, h) {
  const rgba = wctx.getImageData(0, 0, w, h).data;
  const t0 = performance.now();
  const lb = letterbox(rgba, w, h);
  const det = await detSession.run({ images: lb.tensor });
  let found = decode(det[detSession.outputNames[0]], lb, w, h)
    .filter((f) => Math.trunc(f.box[2]) > Math.trunc(f.box[0]) && Math.trunc(f.box[3]) > Math.trunc(f.box[1]))
    .slice(0, MAX_FACES);
  if (found.length) {
    const out = await emoSession.run({ input: emotionBatch(rgba, w, found) });
    const logits = out[emoSession.outputNames[0]].data, k = EMOTIONS.length;
    found.forEach((f, i) => {
      f.probs = softmax(Array.from(logits.slice(i * k, (i + 1) * k)));
      f.top = f.probs.indexOf(Math.max(...f.probs));
    });
  }
  return { faces: found, ms: performance.now() - t0 };
}

function largest(list) {
  let best = -1, area = -1;
  list.forEach((f, i) => {
    const a = (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]);
    if (a > area) { area = a; best = i; }
  });
  return best;
}

// ---------- drawing ----------
function render() {
  if (!current) return;
  canvas.width = current.w; canvas.height = current.h;
  ctx.drawImage(work, 0, 0);
  const lw = Math.max(2, Math.round(Math.max(current.w, current.h) / 320));
  const fs = Math.max(14, Math.round(Math.max(current.w, current.h) / 32));
  ctx.font = `600 ${fs}px Inter, system-ui, sans-serif`;
  faces.forEach((f, i) => {
    const [x1, y1, x2, y2] = f.box;
    const color = i === selected ? "#4f46e5" : "#22c55e";
    ctx.lineWidth = lw; ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = `${EMOTIONS[f.top]} ${(f.probs[f.top] * 100).toFixed(0)}%`;
    const tw = ctx.measureText(label).width + 10, th = fs + 8;
    const ty = y1 - th < 0 ? y2 : y1 - th;
    ctx.fillStyle = color; ctx.fillRect(x1 - lw / 2, ty, tw, th);
    ctx.fillStyle = "#fff"; ctx.fillText(label, x1 + 5, ty + fs);
  });
  renderBars();
}

function renderBars() {
  if (!faces.length) {
    $("faces").innerHTML = "";
    $("bars").innerHTML = `<div class="placeholder">No face found. Try a clearer, front-facing photo.</div>`;
    return;
  }
  $("result-sub").textContent = `${faces.length} face${faces.length > 1 ? "s" : ""} found. Showing face ${selected + 1}.`;
  $("faces").innerHTML = faces.length > 1 ? faces.map((f, i) =>
    `<button class="face-chip${i === selected ? " active" : ""}" data-i="${i}">Face ${i + 1}: ${EMOTIONS[f.top]}</button>`).join("") : "";
  const f = faces[selected];
  $("bars").innerHTML = EMOTIONS.map((e, i) => {
    const p = f.probs[i] * 100;
    return `<div class="emo-row${i === f.top ? " top" : ""}"><span>${e}</span>
      <div class="bar"><span style="width:${p.toFixed(1)}%"></span></div><span class="pct">${p.toFixed(1)}%</span></div>`;
  }).join("") + `<p class="muted">Detector confidence: ${(f.score * 100).toFixed(0)}%</p>`;
}

$("faces").addEventListener("click", (e) => {
  const b = e.target.closest(".face-chip");
  if (b) { selected = Number(b.dataset.i); render(); }
});

canvas.addEventListener("click", (e) => {
  if (!faces.length) return;
  const r = canvas.getBoundingClientRect();
  const x = (e.clientX - r.left) * canvas.width / r.width, y = (e.clientY - r.top) * canvas.height / r.height;
  const hit = faces.findIndex((f) => x >= f.box[0] && x <= f.box[2] && y >= f.box[1] && y <= f.box[3]);
  if (hit >= 0) { selected = hit; render(); }
});

function setHud(items) {
  $("hud").innerHTML = items.map(([k, v]) => `<span>${k}: <b>${v}</b></span>`).join("");
}

// ---------- photo mode ----------
async function runOnImage(img) {
  stopCam();
  let w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
  const s = Math.min(1, MAX_SIDE / Math.max(w, h));
  w = Math.round(w * s); h = Math.round(h * s);
  work.width = w; work.height = h;
  wctx.drawImage(img, 0, 0, w, h);
  current = { w, h };
  $("stage-msg").hidden = true;
  const res = await analyze(w, h);
  faces = res.faces; selected = largest(faces);
  render();
  setHud([["Faces", faces.length], ["Inference", `${res.ms.toFixed(0)} ms`], ["Image", `${w}×${h}`]]);
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

$("file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  document.querySelectorAll(".sample").forEach((b) => b.classList.remove("active"));
  try { await runOnImage(await loadImage(url)); }
  catch { $("stage-msg").hidden = false; $("stage-msg").textContent = "Could not read that image."; }
  finally { URL.revokeObjectURL(url); e.target.value = ""; }
});

// ---------- webcam mode ----------
async function startCam() {
  const video = $("video");
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }, audio: false });
  } catch (err) {
    $("stage-msg").hidden = false;
    $("stage-msg").textContent = "Camera not available: " + err.message;
    return;
  }
  video.srcObject = stream;
  await video.play();
  $("cam").textContent = "Stop webcam";
  $("cam").classList.add("danger");
  $("stage-msg").hidden = true;
  document.querySelectorAll(".sample").forEach((b) => b.classList.remove("active"));
  running = true;
  let fps = 0, last = performance.now();
  const loop = async () => {
    if (!running) return;
    const w = video.videoWidth, h = video.videoHeight;
    if (w && h) {
      if (work.width !== w || work.height !== h) { work.width = w; work.height = h; }
      wctx.save(); wctx.translate(w, 0); wctx.scale(-1, 1);       // mirror like a selfie camera
      wctx.drawImage(video, 0, 0, w, h); wctx.restore();
      current = { w, h };
      const res = await analyze(w, h);
      if (!running) return;
      const prevTop = selected >= 0 && faces[selected] ? faces[selected] : null;
      faces = res.faces;
      selected = prevTop && faces.length ? nearest(prevTop, faces) : largest(faces);
      render();
      const now = performance.now();
      fps = fps ? 0.85 * fps + 0.15 * (1000 / (now - last)) : 1000 / (now - last);
      last = now;
      setHud([["FPS", fps.toFixed(1)], ["Inference", `${res.ms.toFixed(0)} ms`], ["Faces", faces.length]]);
    }
    requestAnimationFrame(loop);
  };
  loop();
}

function nearest(prev, list) {
  let best = 0, score = -1;
  list.forEach((f, i) => { const v = iou(prev.box, f.box); if (v > score) { score = v; best = i; } });
  return score > 0.1 ? best : largest(list);
}

function stopCam() {
  running = false;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  $("cam").textContent = "Start webcam";
  $("cam").classList.remove("danger");
}

$("cam").addEventListener("click", () => (running ? stopCam() : startCam()));

// ---------- static sections ----------
function fill() {
  $("lead").textContent = meta.subtitle;
  $("chips").innerHTML = meta.tags.map((t) => `<span class="chip">${t}</span>`).join("");
  $("links").innerHTML = [
    `<a class="btn" href="${meta.repo_url}" target="_blank" rel="noopener">GitHub repo</a>`,
    `<a class="btn" href="${meta.notebook_url}" target="_blank" rel="noopener">Notebook</a>`,
  ].join("");
  $("steps").innerHTML = meta.steps.map((s) => `<li>${s}</li>`).join("");
  $("footer").innerHTML = meta.footer;
  $("size-note").textContent = meta.input_size;

  const t = meta.verification_total;
  const onnxRows = meta.verification.filter((v) => v.reference === "ultralytics .onnx");
  const maxDiff = Math.max(...meta.verification.map((v) => v.max_prob_diff));
  const minIou = Math.min(...meta.verification.filter((v) => v.matched).map((v) => v.min_iou));
  const totalMb = Object.values(meta.model_sizes_mb).reduce((a, b) => a + b, 0);
  $("metrics").innerHTML = [
    [`${t.matched}/${t.ref_faces}`, "reference faces matched"],
    [`${t.same_label}/${t.matched}`, "same emotion label"],
    [minIou.toFixed(3), "lowest box IoU"],
    [maxDiff.toFixed(3), "max probability difference"],
    [`${totalMb.toFixed(1)} MB`, "models download"],
  ].map(([b, s]) => `<div class="metric"><b>${b}</b><span>${s}</span></div>`).join("");

  $("verify").innerHTML = `<tr><th>Image</th><th>Reference</th><th>Faces (ref / web)</th><th>Min IoU</th><th>Same emotion</th><th>Max prob. diff</th></tr>` +
    meta.verification.map((v) => `<tr><td>${v.image}</td><td>${v.reference}</td><td>${v.ref_faces} / ${v.ours_faces}</td>
      <td>${v.matched ? v.min_iou.toFixed(3) : "–"}</td><td>${v.same_label}/${v.matched}</td><td>${v.max_prob_diff.toFixed(3)}</td></tr>`).join("");
  void onnxRows;

  $("bench").innerHTML = `<tr><th>Pipeline</th><th>Input</th><th>ms / frame</th><th>FPS</th><th>Faces</th></tr>` +
    meta.benchmark.map((b) => `<tr><td>${b.pipeline}</td><td>${b.size}px</td><td>${b.ms.toFixed(0)}</td>
      <td>${(1000 / b.ms).toFixed(1)}</td><td>${b.faces}</td></tr>`).join("");

  $("samples").innerHTML = meta.samples.map((s, i) =>
    `<button class="sample" data-src="${s.file}" title="${s.credit}" aria-label="Sample: ${s.label}"><img src="${s.file}" alt="${s.label}" loading="lazy"></button>`).join("");
  $("samples").addEventListener("click", async (e) => {
    const b = e.target.closest(".sample");
    if (!b || !detSession) return;
    document.querySelectorAll(".sample").forEach((x) => x.classList.toggle("active", x === b));
    await runOnImage(await loadImage(b.dataset.src));
  });
}

async function init() {
  meta = await (await fetch("meta.json")).json();
  SIZE = meta.input_size; CONF = meta.conf; IOU = meta.iou; EMOTIONS = meta.emotions;
  fill();
  ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1);
  const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
  try {
    [detSession, emoSession] = await Promise.all([
      ort.InferenceSession.create("models/face_detector.onnx", opts),
      ort.InferenceSession.create("models/emotion.onnx", opts),
    ]);
  } catch (err) {
    $("stage-msg").textContent = "Could not load the models: " + err.message;
    return;
  }
  $("cam").disabled = false;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    $("cam").disabled = true; $("cam").title = "Camera needs HTTPS and a supported browser";
  }
  const first = document.querySelector(".sample");
  first.classList.add("active");
  await runOnImage(await loadImage(first.dataset.src));
}

init();

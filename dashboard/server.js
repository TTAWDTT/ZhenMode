// ocean-solver cluster dashboard server.
//
// Spawns one persistent python cluster.py bridge (gateway PTY) and polls
// every registered run from runs.json: last progress row, liveness, verdict.
// Serves a single-page UI (public/) with run table, deepT/AMOC charts
// (Chart.js), and live log tail.
//
//   npm start          → http://localhost:8399
"use strict";

const { spawn } = require("child_process");
const express = require("express");
const fs = require("fs");
const path = require("path");

const HERE = __dirname;
const PORT = 8399;
const POLL_MS = 30_000;        // run-status poll
const CURVE_TTL_MS = 3 * 60_000;   // analysis npz re-fetch
const ANALYSIS_MIN_INTERVAL_MS = 5 * 60_000; // cluster-side recompute cadence
const LOG_TTL_MS = 10_000;     // log tail cache

// ── registry ──
const runsJson = () =>
  JSON.parse(fs.readFileSync(path.join(HERE, "runs.json"), "utf-8"));

// ── python bridge (one per node set in secret.json) ──
let bridge = null;
let bridgeSeq = 1;
const pending = new Map(); // id → {resolve, reject, timer}

function startBridge() {
  bridge = spawn("python", [path.join(HERE, "cluster.py")], {
    stdio: ["pipe", "pipe", "inherit"],
  });
  let buf = "";
  bridge.stdout.setEncoding("utf-8");
  bridge.stdout.on("data", (chunk) => {
    buf += chunk;
    let idx;
    while ((idx = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 1);
      if (!line) continue;
      try {
        const msg = JSON.parse(line);
        const p = pending.get(msg.id);
        if (p) {
          clearTimeout(p.timer);
          pending.delete(msg.id);
          if (msg.ok) p.resolve(msg.out);
          else p.reject(new Error(msg.err));
        }
      } catch { /* partial line */ }
    }
  });
  bridge.on("exit", (code) => {
    console.error(`[bridge] exited code=${code}, respawning in 5s`);
    for (const [id, p] of pending) {
      clearTimeout(p.timer);
      p.reject(new Error("bridge exited"));
      pending.delete(id);
    }
    setTimeout(startBridge, 5000);
  });
}

function clusterCmd(cmd, wait = 10) {
  return new Promise((resolve, reject) => {
    if (!bridge || bridge.exitCode !== null) {
      return reject(new Error("bridge not running"));
    }
    const id = bridgeSeq++;
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new Error(`cluster cmd timeout: ${cmd.slice(0, 60)}`));
    }, (wait + 60) * 1000);
    pending.set(id, { resolve, reject, timer });
    bridge.stdin.write(JSON.stringify({ cmd, id, wait }) + "\n");
  });
}

// ── state cache ──
const state = new Map(); // tag → {row, alive, verdict, warn, ts}
let lastPoll = 0;

function stripAnsi(s) {
  // eslint-disable-next-line no-control-regex
  return s.replace(/\x1b\[[0-9;]*m/g, "").replace(/[\x00-\x09\x0b-\x1f]/g, "");
}

async function pollRun(run) {
  const q =
    `grep VERDICT ${run.log} 2>/dev/null | tail -1 | sed 's/\\x1b\\[[0-9;]*m//g'; ` +
    `grep -E '^\\s*[0-9]+\\.[0-9]' ${run.log} | tail -1; ` +
    `pgrep -fc 'tag ${run.tag}' || echo 0`;
  const out = stripAnsi(await clusterCmd(q, 12));
  const lines = out.split("\n").map((l) => l.trim())
    .filter((l) => l && !l.endsWith("~]") && !l.includes("root@"));
  let verdict = null, row = "", alive = false;
  for (const l of lines) {
    if (l.includes("VERDICT")) {
      verdict = l.includes("PASS") ? "PASS" : "FAIL";
    } else if (/^\d+(\.\d+)?\s+\d+\s/.test(l)) {
      row = l;
    } else if (/^\d+$/.test(l)) {
      alive = parseInt(l, 10) > 0;
    }
  }
  const f = row.split(/\s+/);
  const st = {
    tag: run.tag,
    node: run.node,
    gpu: run.gpu,
    total_days: run.total_days,
    started: run.started,
    note: run.note || "",
    alive,
    verdict,
    day: f[0] ? parseFloat(f[0]) : null,
    max_u: f[2] ? parseFloat(f[2]) : null,
    max_T: f[3] ? parseFloat(f[3]) : null,
    max_eta: f[4] ? parseFloat(f[4]) : null,
    nan: f[7] ? parseInt(f[7], 10) : null,
    warn: [],
    row,
    ts: Date.now(),
  };
  if (st.max_u > 5) st.warn.push(`max|u|=${st.max_u}`);
  if (st.max_eta > 10) st.warn.push(`max|eta|=${st.max_eta}m`);
  if (st.nan > 0) st.warn.push(`NaN=${st.nan}`);
  return st;
}

async function pollAll() {
  const runs = runsJson().runs;
  for (const run of runs) {
    try {
      state.set(run.tag, await pollRun(run));
    } catch (e) {
      const prev = state.get(run.tag) || {};
      state.set(run.tag, {
        ...prev, tag: run.tag, node: run.node, gpu: run.gpu,
        total_days: run.total_days, started: run.started, note: run.note,
        error: String(e.message || e), ts: Date.now(),
      });
    }
  }
  lastPoll = Date.now();
}

// ── curves: fetch analysis npz from node, decode via python one-shot ──
// The npz is produced by results/_spinup_probe_analysis.py on the cluster.
// For a live run we trigger a detached recompute (mtime-gated on the newest
// 3D snap) whenever the cache is stale — curves then advance on their own.
const curveCache = new Map(); // tag → {data, ts}
const lastAnalysisKick = new Map(); // tag → ts

function clusterHasRun(tag) {
  const st = state.get(tag);
  return st && st.alive && !st.verdict;
}

async function kickAnalysis(tag) {
  const last = lastAnalysisKick.get(tag) || 0;
  if (Date.now() - last < ANALYSIS_MIN_INTERVAL_MS) return; // rate-limit
  lastAnalysisKick.set(tag, Date.now());
  // nohup + mtime gate: skip if no snap newer than the existing analysis npz
  const cmd =
    `d=/data/tmp/ocean/results/global_${tag}_3d; a=/data/tmp/ocean/results/${tag}_analysis.npz; ` +
    `n=$(ls -t $d/snap_*.npy 2>/dev/null | head -1); ` +
    `if [ -n "$n" ] && { [ ! -f $a ] || [ $n -nt $a ]; }; then ` +
    `nohup docker exec -e PARENT_NPZ=/data/tmp/ocean/results/global_spinA30probe.npz ` +
    `-w /data/tmp/ocean jaxtest2 /opt/conda/envs/py/bin/python ` +
    `results/_spinup_probe_analysis.py ${tag} > /data/tmp/ocean/logs/analysis_${tag}.log 2>&1 & fi; echo kicked`;
  try { await clusterCmd(cmd, 8); } catch { /* retry next TTL */ }
}

async function getCurves(tag) {
  const hit = curveCache.get(tag);
  if (hit) {
    if (Date.now() - hit.ts < CURVE_TTL_MS) return hit.data;
    if (clusterHasRun(tag)) kickAnalysis(tag); // refresh source, don't block
  }
  const b64raw = await clusterCmd(
    `base64 -w0 /data/tmp/ocean/results/${tag}_analysis.npz`, 30);
  // PTY output can carry the command echo + trailing shell prompt around the
  // payload; pull out the longest contiguous base64 run as the payload.
  const m = b64raw.match(/[A-Za-z0-9+/=]{100,}/);
  if (!m) throw new Error("no base64 payload in cluster output");
  const b64 = m[0];
  const py = spawn("python", ["-c", `
import sys, json, base64, io, numpy as np
raw = sys.stdin.read().strip()
d = np.load(io.BytesIO(base64.b64decode(raw)))
out = {k: d[k].tolist() for k in d.files if d[k].ndim == 1}
print(json.dumps(out))`], {
    stdio: ["pipe", "pipe", "inherit"],
  });
  const data = await new Promise((resolve, reject) => {
    let buf = "";
    py.stdout.on("data", (c) => (buf += c));
    py.on("exit", (code) =>
      code === 0
        ? resolve(JSON.parse(buf))
        : reject(new Error(`npz decode failed code=${code}`)));
    py.stdin.write(b64);
    py.stdin.end();
  });
  data.ts = Date.now();           // client-side "updated at" display
  curveCache.set(tag, { data, ts: Date.now() });
  return data;
}

// ── log tail ──
const logCache = new Map(); // tag → {text, ts}

async function getLogTail(tag, bytes = 4000) {
  const hit = logCache.get(tag);
  if (hit && Date.now() - hit.ts < LOG_TTL_MS) return hit.text;
  const run = runsJson().runs.find((r) => r.tag === tag);
  if (!run) throw new Error("unknown tag");
  const out = stripAnsi(
    await clusterCmd(`tail -c ${bytes} ${run.log}`, 10));
  logCache.set(tag, { text: out, ts: Date.now() });
  return out;
}

// ── app ──
const app = express();
app.use(express.static(path.join(HERE, "public")));

app.get("/api/overview", async (_req, res) => {
  if (Date.now() - lastPoll > POLL_MS) await pollAll();
  res.json({ runs: [...state.values()], lastPoll });
});

app.get("/api/curves/:tag", async (req, res) => {
  try {
    res.json(await getCurves(req.params.tag));
  } catch (e) {
    res.status(502).json({ error: String(e.message || e) });
  }
});

app.get("/api/log/:tag", async (req, res) => {
  try {
    res.type("text/plain").send(await getLogTail(req.params.tag));
  } catch (e) {
    res.status(502).json({ error: String(e.message || e) });
  }
});

app.get("/api/pollnow", async (_req, res) => {
  await pollAll();
  res.json({ ok: true });
});

startBridge();
setTimeout(pollAll, 1500);          // first poll after bridge warm-up
setInterval(pollAll, POLL_MS);

app.listen(PORT, () =>
  console.log(`[dashboard] http://localhost:${PORT}  (poll ${POLL_MS / 1000}s)`));

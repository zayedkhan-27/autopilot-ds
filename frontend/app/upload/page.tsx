"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import AppShell from "../../components/AppShell";
import { api } from "../../lib/api";

export default function UploadPage() {
  const router = useRouter();
  const [file,setFile] = useState<File|null>(null);
  const [target,setTarget] = useState("churn");
  const [task,setTask] = useState("classification");
  const [iterations,setIterations] = useState(3);
  const [status,setStatus] = useState("IDLE");
  const [meta,setMeta] = useState<any>(null);
  const [error,setError] = useState("");

  async function launch() {
    if (!file) return;
    setError("");
    try {
      setStatus("UPLOADING");
      const ds = await api.uploadDataset(file);
      setMeta(ds);
      setStatus("RUNNING");
      const exp = await api.createExperiment({
        dataset_id: ds.id,
        target_column: target || null,
        task_type: task || null,
        max_iterations: iterations,
      });
      setStatus("COMPLETE");
      router.push(`/experiments/${exp.id}`);
    } catch (err:any) {
      setStatus("FAILED");
      setError(err.message || "Mission failed");
    }
  }

  return (
    <AppShell>
      <div className="page-kicker">NEW MISSION / INGEST</div>
      <div className="dash-head">
        <h1>FEED THE<br/><i>ENGINE.</i></h1>
        <div className={`run-state ${status.toLowerCase()}`}>{status} ●</div>
      </div>

      <div className="upload-grid">
        <label className="dropzone">
          <input type="file" accept=".csv,.xlsx" onChange={e=>setFile(e.target.files?.[0]||null)} />
          <span className="drop-icon">↥</span>
          <strong>{file ? file.name : "DROP DATASET HERE"}</strong>
          <p>{file ? `${(file.size/1024/1024).toFixed(2)} MB` : "CSV or XLSX / max 50 MB"}</p>
          <em>CHOOSE FILE</em>
        </label>

        <div className="config-card">
          <div className="form-code">MISSION / CONFIGURATION</div>
          <label>TARGET COLUMN<input value={target} onChange={e=>setTarget(e.target.value)} placeholder="e.g. churn"/></label>
          <label>TASK TYPE
            <select value={task} onChange={e=>setTask(e.target.value)}>
              <option value="classification">Classification</option>
              <option value="regression">Regression</option>
            </select>
          </label>
          <label>SELF-CORRECTION ITERATIONS
            <select value={iterations} onChange={e=>setIterations(Number(e.target.value))}>
              {[1,2,3,4,5].map(n=><option key={n} value={n}>{n}</option>)}
            </select>
          </label>
          {meta && <div className="mini-meta">{meta.rows} ROWS / {meta.columns} COLUMNS / DATASET #{meta.id}</div>}
          {error && <div className="error-box">{error}</div>}
          <button disabled={!file || status==="UPLOADING" || status==="RUNNING"} className="primary-btn full" onClick={launch}>
            {status==="RUNNING" ? "AUTOPILOT RUNNING..." : status==="UPLOADING" ? "UPLOADING..." : "ENGAGE AUTOPILOT →"}
          </button>
        </div>
      </div>

      <section className="pipeline-console">
        {["PROFILE","CLEAN","ENGINEER","TRAIN","VERIFY"].map((s,i)=>(
          <div className="console-step" key={s}><span>0{i+1}</span><strong>{s}</strong><em>{status==="RUNNING"?"◌":"○"}</em></div>
        ))}
      </section>
    </AppShell>
  );
}

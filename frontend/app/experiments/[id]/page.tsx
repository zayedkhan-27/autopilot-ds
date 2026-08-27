"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import AppShell from "../../../components/AppShell";
import { api } from "../../../lib/api";

function pct(v:any) {
  const n = Number(v);
  if (Number.isNaN(n)) return "—";
  return n <= 1 ? `${(n*100).toFixed(1)}%` : n.toFixed(2);
}

export default function ExperimentDetail() {
  const params = useParams();
  const id = Number(params.id);
  const [data,setData] = useState<any>(null);
  const [error,setError] = useState("");

  useEffect(()=>{ if(id) api.experiment(id).then(setData).catch(e=>setError(e.message)); },[id]);

  const result = data?.result || {};
  const health = result.pipeline_health || {};
  const metrics = result.metrics || {};
  const modelEntries = Object.entries(metrics);

  return (
    <AppShell>
      <div className="page-kicker">MISSION #{String(id).padStart(3,"0")} / RESULTS</div>
      <div className="dash-head">
        <h1>{data?.status==="COMPLETED"?"MISSION":"MISSION"}<br/><i>{data?.status || "LOADING"}.</i></h1>
        <Link className="ghost-btn dark" href="/upload">NEW MISSION +</Link>
      </div>
      {error && <div className="error-box">{error}</div>}
      {data?.error_message && <div className="error-box">{data.error_message}</div>}

      <div className="stats">
        <div className="stat accent"><span>BEST MODEL</span><strong className="model-name">{result.best_model_name || "—"}</strong><small>selected champion</small></div>
        <div className="stat"><span>TASK</span><strong className="model-name">{result.task_type || "—"}</strong><small>{result.target_column || "target auto"}</small></div>
        <div className="stat"><span>HEALTH</span><strong>{health.overall_score != null ? Number(health.overall_score).toFixed(0) : "—"}</strong><small>pipeline score</small></div>
        <div className="stat"><span>ITERATION</span><strong>{result.iteration ?? "—"}</strong><small>self-correction</small></div>
      </div>

      <section className="panel">
        <div className="panel-head"><h2>MODEL ARENA</h2><span>{modelEntries.length} CONTENDERS</span></div>
        {modelEntries.length===0 ? <div className="empty">NO MODEL METRICS AVAILABLE.</div> :
          modelEntries.map(([name,m]:any)=>{
            const score = m.accuracy ?? m.f1 ?? m.r2 ?? m.rmse ?? m.score;
            return <div className={`model-row ${name===result.best_model_name?"winner":""}`} key={name}>
              <div><span>{name===result.best_model_name?"CHAMPION":"CONTENDER"}</span><strong>{name}</strong></div>
              <div className="metric-stack">
                {Object.entries(m).filter(([_,v])=>typeof v==="number").slice(0,4).map(([k,v])=>
                  <span key={k}><small>{k.toUpperCase()}</small><b>{pct(v)}</b></span>
                )}
              </div>
              <strong className="big-score">{pct(score)}</strong>
            </div>
          })
        }
      </section>

      <div className="result-grid">
        <section className="panel">
          <div className="panel-head"><h2>CLEANING LOG</h2></div>
          <pre className="json-view">{JSON.stringify(result.cleaning_log || {}, null, 2)}</pre>
        </section>
        <section className="panel">
          <div className="panel-head"><h2>VERIFICATION</h2></div>
          <pre className="json-view">{JSON.stringify(result.verification_report || {}, null, 2)}</pre>
        </section>
      </div>
    </AppShell>
  );
}

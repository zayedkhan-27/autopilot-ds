"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import AppShell from "../../components/AppShell";
import { api, Experiment } from "../../lib/api";

export default function ExperimentsPage() {
  const [rows,setRows] = useState<Experiment[]>([]);
  const [error,setError] = useState("");

  useEffect(()=>{ api.experiments().then(setRows).catch(e=>setError(e.message)); },[]);

  return (
    <AppShell>
      <div className="page-kicker">ARCHIVE / EXPERIMENTS</div>
      <div className="dash-head"><h1>MISSION<br/><i>ARCHIVE.</i></h1><Link className="primary-btn" href="/upload">NEW MISSION +</Link></div>
      {error && <div className="error-box">{error}</div>}
      <section className="panel">
        <div className="panel-head"><h2>ALL RUNS</h2><span>{rows.length} RECORDS</span></div>
        {rows.length===0 ? <div className="empty">NO EXPERIMENTS FOUND.</div> :
          rows.map(x=>(
            <Link href={`/experiments/${x.id}`} className="mission-row" key={x.id}>
              <span className={`badge ${x.status.toLowerCase()}`}>{x.status}</span>
              <strong>MISSION #{String(x.id).padStart(3,"0")}</strong>
              <span>{x.task_type || "AUTO"}</span>
              <span>{x.best_model_name || "—"}</span>
              <b>{x.health_score != null ? `${Number(x.health_score).toFixed(0)}/100` : "—"}</b>
              <em>↗</em>
            </Link>
          ))
        }
      </section>
    </AppShell>
  );
}

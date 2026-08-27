"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import AppShell from "../../components/AppShell";
import { api, Dataset, Experiment, User } from "../../lib/api";

export default function Dashboard() {
  const router = useRouter();
  const [user,setUser] = useState<User|null>(null);
  const [datasets,setDatasets] = useState<Dataset[]>([]);
  const [experiments,setExperiments] = useState<Experiment[]>([]);
  const [error,setError] = useState("");

  useEffect(() => {
    Promise.all([api.me(), api.datasets(), api.experiments()])
      .then(([u,d,e]) => {setUser(u);setDatasets(d);setExperiments(e);})
      .catch(err => {
        setError(err.message);
        if ((err.message||"").toLowerCase().includes("token")) router.push("/login");
      });
  }, [router]);

  const complete = experiments.filter(x=>x.status==="COMPLETED");
  const best = complete.reduce((m,x)=>Math.max(m, Number(x.health_score||0)),0);

  return (
    <AppShell>
      <div className="page-kicker">MISSION CONTROL / OVERVIEW</div>
      <div className="dash-head">
        <h1>GOOD TO SEE YOU,<br/><i>{user?.name?.toUpperCase() || "OPERATOR"}.</i></h1>
        <Link className="primary-btn" href="/upload">NEW MISSION +</Link>
      </div>

      {error && <div className="error-box">{error}</div>}

      <div className="stats">
        <div className="stat"><span>DATASETS</span><strong>{datasets.length}</strong><small>stored inputs</small></div>
        <div className="stat"><span>EXPERIMENTS</span><strong>{experiments.length}</strong><small>total missions</small></div>
        <div className="stat"><span>COMPLETED</span><strong>{complete.length}</strong><small>verified runs</small></div>
        <div className="stat accent"><span>BEST HEALTH</span><strong>{best ? best.toFixed(0) : "—"}</strong><small>pipeline score</small></div>
      </div>

      <section className="panel">
        <div className="panel-head"><h2>RECENT MISSIONS</h2><Link href="/experiments">VIEW ALL ↗</Link></div>
        {experiments.length===0 ? (
          <div className="empty">NO MISSIONS YET. <Link href="/upload">LAUNCH ONE →</Link></div>
        ) : experiments.slice(0,6).map(x=>(
          <Link href={`/experiments/${x.id}`} className="mission-row" key={x.id}>
            <span className={`badge ${x.status.toLowerCase()}`}>{x.status}</span>
            <strong>MISSION #{String(x.id).padStart(3,"0")}</strong>
            <span>{x.task_type || "AUTO"}</span>
            <span>{x.best_model_name || "—"}</span>
            <b>{x.health_score != null ? `${Number(x.health_score).toFixed(0)}/100` : "—"}</b>
            <em>↗</em>
          </Link>
        ))}
      </section>
    </AppShell>
  );
}

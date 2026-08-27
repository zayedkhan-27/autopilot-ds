"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import TopBar from "../components/TopBar";

const stages = [
  ["01","PROFILE","Map schema, missing values, cardinality and data quality risks."],
  ["02","CLEAN","Repair missing values, duplicates and inconsistent data."],
  ["03","ENGINEER","Build reproducible model-ready feature transformations."],
  ["04","TRAIN","Challenge multiple algorithms against the same dataset."],
  ["05","VERIFY","Independently validate model quality and pipeline health."],
];

export default function Home() {
  return (
    <main className="landing">
      <TopBar />
      <section className="hero">
        <motion.div initial={{opacity:0,y:15}} animate={{opacity:1,y:0}} className="eyebrow">
          AUTONOMOUS DATA INTELLIGENCE / 01
        </motion.div>
        <motion.h1 initial={{opacity:0,y:40}} animate={{opacity:1,y:0}} transition={{delay:.08}}>
          YOUR DATA.<br/><i>ON AUTOPILOT.</i>
        </motion.h1>
        <p className="hero-copy">
          Upload raw data. AutoPilot profiles, cleans, engineers, trains,
          verifies and explains the best machine-learning pipeline.
        </p>
        <div className="hero-actions">
          <Link className="primary-btn" href="/register">START A MISSION <span>→</span></Link>
          <a className="ghost-btn" href="#engine">EXPLORE ENGINE ↓</a>
        </div>
        <div className="hero-index">AP / 001</div>
      </section>

      <section id="engine" className="engine-section">
        <div className="section-kicker">THE ENGINE / LIVE PIPELINE</div>
        {stages.map(([n,title,desc]) => (
          <div className="engine-row" key={title}>
            <span className="engine-num">{n}</span>
            <div>
              <h2>{title}</h2>
              <p>{desc}</p>
            </div>
            <span className="status-dot">○</span>
          </div>
        ))}
      </section>

      <section className="statement">
        <div className="section-kicker">BUILT FOR DECISIONS / NOT DEMOS</div>
        <h2>FROM MESSY CSV<br/>TO <i>TRUSTED MODEL.</i></h2>
        <p>
          Every mission stores the dataset, selected task, model results and
          pipeline health — so your experiments remain reproducible.
        </p>
        <Link className="primary-btn" href="/register">ENTER WORKSPACE <span>↗</span></Link>
      </section>
    </main>
  );
}

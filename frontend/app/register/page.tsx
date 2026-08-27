"use client";

import { FormEvent, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import TopBar from "../../components/TopBar";
import { register } from "../../lib/api";

export default function RegisterPage() {
  const router = useRouter();
  const [name,setName] = useState("");
  const [email,setEmail] = useState("");
  const [password,setPassword] = useState("");
  const [error,setError] = useState("");
  const [loading,setLoading] = useState(false);

  async function submit(e:FormEvent) {
    e.preventDefault();
    setError(""); setLoading(true);
    try {
      await register(name,email,password);
      router.push("/dashboard");
    } catch (err:any) {
      setError(err.message || "Registration failed");
    } finally { setLoading(false); }
  }

  return (
    <main className="auth-page">
      <TopBar />
      <div className="auth-wrap">
        <section className="auth-title">
          <div className="section-kicker">NEW OPERATOR / 01</div>
          <h1>TAKE<br/><i>CONTROL.</i></h1>
          <p>Create your workspace and launch your first AutoPilot mission.</p>
        </section>
        <form className="auth-card" onSubmit={submit}>
          <div className="form-code">AUTH / REGISTER</div>
          <label>NAME<input value={name} onChange={e=>setName(e.target.value)} required placeholder="Your name"/></label>
          <label>EMAIL<input value={email} onChange={e=>setEmail(e.target.value)} type="email" required placeholder="you@example.com"/></label>
          <label>PASSWORD<input value={password} onChange={e=>setPassword(e.target.value)} type="password" minLength={8} required placeholder="Minimum 8 characters"/></label>
          {error && <div className="error-box">{error}</div>}
          <button className="primary-btn full" disabled={loading}>{loading?"CREATING WORKSPACE...":"CREATE WORKSPACE →"}</button>
          <p className="auth-switch">Already registered? <Link href="/login">Log in ↗</Link></p>
        </form>
      </div>
    </main>
  );
}

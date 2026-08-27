"use client";

import { FormEvent, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import TopBar from "../../components/TopBar";
import { login } from "../../lib/api";

export default function LoginPage() {
  const router = useRouter();
  const [email,setEmail] = useState("");
  const [password,setPassword] = useState("");
  const [error,setError] = useState("");
  const [loading,setLoading] = useState(false);

  async function submit(e:FormEvent) {
    e.preventDefault();
    setError(""); setLoading(true);
    try {
      await login(email,password);
      router.push("/dashboard");
    } catch (err:any) {
      setError(err.message || "Login failed");
    } finally { setLoading(false); }
  }

  return (
    <main className="auth-page">
      <TopBar />
      <div className="auth-wrap">
        <section className="auth-title">
          <div className="section-kicker">SECURE ACCESS / 01</div>
          <h1>WELCOME<br/><i>BACK.</i></h1>
          <p>Resume your datasets, experiments and model results.</p>
        </section>
        <form className="auth-card" onSubmit={submit}>
          <div className="form-code">AUTH / LOGIN</div>
          <label>EMAIL<input value={email} onChange={e=>setEmail(e.target.value)} type="email" required placeholder="you@example.com"/></label>
          <label>PASSWORD<input value={password} onChange={e=>setPassword(e.target.value)} type="password" required placeholder="••••••••"/></label>
          {error && <div className="error-box">{error}</div>}
          <button className="primary-btn full" disabled={loading}>{loading?"AUTHENTICATING...":"LOG IN →"}</button>
          <p className="auth-switch">No account? <Link href="/register">Create one ↗</Link></p>
        </form>
      </div>
    </main>
  );
}

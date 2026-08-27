"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { logout } from "../lib/api";

export default function TopBar({app = false}:{app?: boolean}) {
  const router = useRouter();
  return (
    <header className="topbar">
      <Link className="logo" href="/">AUTO<span>PILOT</span></Link>
      <div className="topbar-actions">
        {app ? (
          <>
            <Link href="/dashboard">MISSION CONTROL</Link>
            <button className="text-btn" onClick={() => { logout(); router.push("/login"); }}>
              LOG OUT ↗
            </button>
          </>
        ) : (
          <>
            <Link href="/login">LOG IN</Link>
            <Link className="pill" href="/register">START AUTOPILOT ↗</Link>
          </>
        )}
      </div>
    </header>
  );
}

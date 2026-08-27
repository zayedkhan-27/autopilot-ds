"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import TopBar from "./TopBar";

const nav = [
  ["/dashboard","OVERVIEW"],
  ["/upload","NEW MISSION"],
  ["/experiments","EXPERIMENTS"],
];

export default function AppShell({children}:{children:React.ReactNode}) {
  const pathname = usePathname();
  return (
    <div className="app-page">
      <TopBar app />
      <div className="app-grid">
        <aside className="side">
          <div className="side-label">AUTOPILOT / OS</div>
          <nav>
            {nav.map(([href,label],i) => (
              <Link key={href} className={pathname===href?"side-link active":"side-link"} href={href}>
                <span>0{i+1}</span>{label}
              </Link>
            ))}
          </nav>
          <div className="side-foot">SYSTEM<br/><strong>ONLINE ●</strong></div>
        </aside>
        <section className="app-main">{children}</section>
      </div>
    </div>
  );
}

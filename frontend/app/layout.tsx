import "./globals.css";

export const metadata = {
  title: "AUTOPILOT — Autonomous Data Intelligence",
  description: "From raw data to trusted predictions.",
};

export default function RootLayout({children}:{children:React.ReactNode}) {
  return <html lang="en"><body>{children}</body></html>;
}

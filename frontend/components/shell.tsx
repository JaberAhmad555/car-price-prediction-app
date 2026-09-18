"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { createContext, useContext, useEffect, useRef, useState } from "react";
import { ArrowUpRight, ArrowRight, CarFront, CheckCircle2, Menu, X } from "lucide-react";

const ToastContext = createContext<(text: string) => void>(() => {});
export const useToast = () => useContext(ToastContext);
const links = [["/", "Home"], ["/estimate", "Price estimator"], ["/inspection", "AI inspection"], ["/insights", "Market insights"], ["/methodology", "Methodology"]];

export function Shell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const [toast, setToast] = useState("");
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => () => { if (timer.current) clearTimeout(timer.current); }, []);
  useEffect(() => {
    if (!open) return;
    const dismiss = (event: KeyboardEvent) => {
      if (event.key === "Escape") { setOpen(false); document.querySelector<HTMLButtonElement>(".mobile-menu")?.focus(); }
    };
    window.addEventListener("keydown", dismiss);
    return () => window.removeEventListener("keydown", dismiss);
  }, [open]);
  function notify(text: string) {
    setToast(text);
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => setToast(""), 4500);
  }
  return <ToastContext.Provider value={notify}>
    <a className="skip-link" href="#main">Skip to content</a>
    <header className="site-header">
      <div className="container nav-wrap">
        <Link href="/" className="brand" aria-label="CarValue BD home"><span className="brand-icon"><CarFront size={23} strokeWidth={1.8} /></span>CarValue<span className="brand-bd">BD</span></Link>
        <nav aria-label="Main navigation" className="desktop-nav">{links.map(([url, label]) => <Link key={url} href={url} aria-current={pathname === url ? "page" : undefined}>{label}</Link>)}</nav>
        <Link className="button button-small nav-cta" href="/estimate">Get a valuation <ArrowUpRight size={16} /></Link>
        <button className="icon-button mobile-menu" onClick={() => setOpen(!open)} aria-expanded={open} aria-controls="mobile-navigation" aria-label={open ? "Close navigation" : "Open navigation"}>{open ? <X /> : <Menu />}</button>
      </div>
      {open && <nav id="mobile-navigation" className="mobile-links" aria-label="Mobile navigation">{links.map(([url, label]) => <Link key={url} href={url} onClick={() => setOpen(false)} aria-current={pathname === url ? "page" : undefined}>{label}<ArrowUpRight size={16} /></Link>)}</nav>}
    </header>
    <main id="main">{children}</main>
    <footer className="footer"><div className="container">
      <div className="footer-top"><div><Link href="/" className="brand"><span className="brand-icon"><CarFront size={22} /></span>CarValue<span className="brand-bd">BD</span></Link><p>Better context. More confident car decisions.<br />Built for Bangladesh.</p></div><div className="footer-links"><Link href="/estimate">Find your car’s value <ArrowRight size={15} /></Link><Link href="/methodology">Explore the methodology <ArrowRight size={15} /></Link><a href="https://github.com/JaberAhmad555/car-price-prediction-app" target="_blank" rel="noreferrer">View on GitHub <ArrowUpRight size={15} /></a></div></div>
      <div className="footer-bottom"><span>© {new Date().getFullYear()} CarValue BD · An independent portfolio project</span><span>Bangladesh market · BDT / ৳ · Historical data</span></div>
    </div></footer>
    {toast && <div className="toast" role="status"><CheckCircle2 size={18} /><span>{toast}</span><button className="icon-button" aria-label="Dismiss notification" onClick={() => setToast("")}><X size={16} /></button></div>}
  </ToastContext.Provider>;
}

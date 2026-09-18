import Link from "next/link";
export default function NotFound() { return <div className="container empty-page"><p className="eyebrow">404 · A different route</p><h1>This road ends here.</h1><p>Let’s get you back to your next car decision.</p><Link className="button" href="/">Back to home</Link></div>; }

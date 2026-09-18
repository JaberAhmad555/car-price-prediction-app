import type { NextConfig } from "next";
import { PHASE_PRODUCTION_BUILD } from "next/constants";
const config: NextConfig = {
  poweredByHeader: false,
  async headers() {
    return [{ source: "/(.*)", headers: [
      { key: "X-Content-Type-Options", value: "nosniff" },
      { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
      { key: "X-Frame-Options", value: "DENY" },
      { key: "Permissions-Policy", value: "camera=(self), microphone=(), geolocation=()" },
    ] }];
  },
};
export default function nextConfig(phase: string): NextConfig {
  if (phase === PHASE_PRODUCTION_BUILD) {
    const api = process.env.NEXT_PUBLIC_API_URL;
    if (!api) throw new Error("Set NEXT_PUBLIC_API_URL to the backend origin before building.");
    const url = new URL(api);
    if (!["http:", "https:"].includes(url.protocol) || url.username || url.password || url.search || url.hash || url.pathname !== "/") {
      throw new Error("NEXT_PUBLIC_API_URL must be an HTTP(S) origin without credentials, path or query.");
    }
    if (process.env.VERCEL && (url.protocol !== "https:" || ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname))) {
      throw new Error("Vercel requires a public HTTPS backend origin.");
    }
  }
  return config;
}

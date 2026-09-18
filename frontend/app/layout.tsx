import type { Metadata } from "next";
import { Shell } from "@/components/shell";
import "@fontsource-variable/manrope";
import "./globals.css";

const deploymentHost = process.env.VERCEL_PROJECT_PRODUCTION_URL || process.env.VERCEL_URL;

export const metadata: Metadata = {
  metadataBase: deploymentHost ? new URL(`https://${deploymentHost}`) : undefined,
  title: { default: "CarValue BD | AI-Powered Bangladesh Vehicle Valuation", template: "%s | CarValue BD" },
  description: "Explore historical Bangladesh used-car market listing values, transparent machine learning, and optional photo-quality inspection. Prices in BDT.",
  applicationName: "CarValue BD",
  keywords: ["Bangladesh car valuation", "used car listing value", "BDT", "CarValue BD"],
  openGraph: {
    title: "CarValue BD | Know What Your Car Is Worth in Bangladesh",
    description: "AI-powered vehicle valuation trained on historical Bangladesh used-car market data. Transparent estimates in BDT.",
    siteName: "CarValue BD", locale: "en_BD", type: "website",
  },
  twitter: { card: "summary_large_image", title: "CarValue BD", description: "AI-Powered Bangladesh Vehicle Valuation" },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body><Shell>{children}</Shell></body></html>;
}

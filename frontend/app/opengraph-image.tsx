import { ImageResponse } from "next/og";

export const alt = "CarValue BD — AI-Powered Bangladesh Vehicle Valuation";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function OpenGraphImage() {
  return new ImageResponse(<div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column", justifyContent: "space-between", background: "#10191f", padding: "64px 76px", color: "#f2f1ec" }}>
    <div style={{ display: "flex", fontSize: 30, color: "#efbb94" }}>CarValue BD</div>
    <div style={{ display: "flex", flexDirection: "column", fontSize: 68, lineHeight: 1.08, letterSpacing: -3 }}><span>Know what your car</span><span>is worth in Bangladesh.</span></div>
    <div style={{ display: "flex", borderTop: "1px solid #394951", paddingTop: 28, fontSize: 22, color: "#bdd9ca" }}>Historical Bangladesh data · Listing values in BDT · Transparent ML</div>
  </div>, size);
}

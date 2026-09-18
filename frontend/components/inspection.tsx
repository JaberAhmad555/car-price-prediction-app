"use client";

import Image from "next/image";
import { useEffect, useRef, useState } from "react";
import { ArrowRight, Camera, Check, CircleAlert, ImagePlus, Info, LoaderCircle, ScanLine, ShieldCheck, Trash2, UploadCloud } from "lucide-react";
import { request, type Inspection } from "@/lib/api";
import { PageIntro } from "./views";
import { useToast } from "./shell";

type Photo = { id: string; file: File; preview: string; result?: Inspection; error?: string };
const views = ["Front", "Rear", "Driver side", "Passenger side", "Front-left angle", "Front-right angle", "Rear-left angle", "Rear-right angle"];

export function PhotoInspection() {
  const [photos, setPhotos] = useState<Photo[]>([]);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  const input = useRef<HTMLInputElement>(null);
  const camera = useRef<HTMLInputElement>(null);
  const previews = useRef(new Set<string>());
  const toast = useToast();
  useEffect(() => { const urls = previews.current; return () => urls.forEach(url => URL.revokeObjectURL(url)); }, []);
  function add(files: FileList | File[]) {
    setError("");
    const accepted: Photo[] = [];
    for (const file of Array.from(files)) {
      if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) { setError("Use JPEG, PNG or WebP. Convert HEIC photos before uploading."); continue; }
      if (file.size > 8 * 1024 * 1024) { setError("Each photo must be smaller than 8 MB."); continue; }
      if (photos.length + accepted.length >= 8) { setError("You can add up to eight photos per inspection."); break; }
      const preview = URL.createObjectURL(file); previews.current.add(preview);
      accepted.push({ id: crypto.randomUUID(), file, preview });
    }
    setPhotos(previous => [...previous, ...accepted]);
    if (accepted.length) toast(`${accepted.length} photo${accepted.length > 1 ? "s" : ""} added. Nothing is uploaded until you run the checks.`);
  }
  function remove(id: string) {
    const photo = photos.find(item => item.id === id);
    if (photo) { URL.revokeObjectURL(photo.preview); previews.current.delete(photo.preview); }
    setPhotos(previous => previous.filter(item => item.id !== id));
  }
  async function analyze() {
    setBusy(true); setError("");
    let successes = 0;
    for (const photo of photos) {
      try {
        const result = await request<Inspection>("/api/inspect", { method: "POST", headers: { "Content-Type": photo.file.type }, body: photo.file });
        setPhotos(previous => previous.map(item => item.id === photo.id ? { ...item, result, error: undefined } : item));
        successes++;
      } catch (reason) {
        setPhotos(previous => previous.map(item => item.id === photo.id ? { ...item, error: reason instanceof Error ? reason.message : "This photo could not be checked." } : item));
      }
    }
    setBusy(false); toast(`${successes} of ${photos.length} photo checks completed.`);
  }
  return <div className="container page-section"><PageIntro eyebrow="INSPECTION LAB · BETA" title="A closer look starts here.">Make your photos inspection-ready. Capture every angle and check lighting, resolution and sharpness before the next step.</PageIntro>
    <div className="notice"><ScanLine size={21} /><div><strong>Photo-quality checks are live. Damage recognition is not.</strong><p>Advanced visible-damage recognition is currently under development. Future versions will analyze visible dents, scratches and exterior damage to complement valuation. Today’s quality checks do not detect damage, verify vehicle presence or apply price deductions.</p></div></div>
    <div className="inspection-layout"><div><div className={`upload-zone ${dragging ? "dragging" : ""}`} onDragOver={event => { event.preventDefault(); if (!busy) setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={event => { event.preventDefault(); setDragging(false); if (!busy) add(event.dataTransfer.files); }}><div className="upload-symbol"><UploadCloud size={34} strokeWidth={1.4} /></div><h2>Your car. Every angle.</h2><p>Drop your photos here, or choose how to add them.</p><div className="upload-actions"><button className="button" onClick={() => input.current?.click()} disabled={busy || photos.length >= 8}><ImagePlus size={17} /> Browse photos</button><button className="button button-ghost" onClick={() => camera.current?.click()} disabled={busy || photos.length >= 8}><Camera size={17} /> Take a photo</button></div><span className="small-muted">JPEG, PNG or WebP · Up to 8 photos · 8 MB each · Maximum 16 MP</span><input ref={input} className="visually-hidden" tabIndex={-1} aria-label="Choose vehicle photos" type="file" accept="image/jpeg,image/png,image/webp" multiple onChange={event => { if (event.target.files) add(event.target.files); event.target.value = ""; }} /><input ref={camera} className="visually-hidden" tabIndex={-1} aria-label="Capture a vehicle photo" type="file" accept="image/jpeg,image/png,image/webp" capture="environment" onChange={event => { if (event.target.files) add(event.target.files); event.target.value = ""; }} /></div>
    {error && <p className="inline-error" role="alert">{error}</p>}
    {photos.length > 0 && <><div className="photo-toolbar"><strong>{photos.length} / 8 photos added</strong><button className="button button-small" onClick={analyze} disabled={busy}>{busy ? <><LoaderCircle size={16} className="spin" /> Checking photos</> : <>Run quality checks <ArrowRight size={16} /></>}</button></div><div className="photo-grid">{photos.map((photo, index) => <article className="photo-card" key={photo.id}><div className="photo-preview"><Image src={photo.preview} alt={`Uploaded vehicle photo ${index + 1}`} fill unoptimized sizes="(max-width: 700px) 100vw, 350px" /><span className="photo-number">PHOTO {String(index + 1).padStart(2, "0")}</span><button className="photo-remove" onClick={() => remove(photo.id)} disabled={busy} aria-label={`Remove photo ${index + 1}`}><Trash2 size={15} /></button></div><div className="photo-details"><strong title={photo.file.name}>{photo.file.name}</strong>{photo.error ? <p className="photo-error" role="alert">{photo.error}</p> : photo.result ? <div className="checks">{photo.result.checks.map(check => <div className={check.passed ? "check-row passed" : "check-row review"} key={check.name}>{check.passed ? <Check size={15} /> : <CircleAlert size={15} />}<div><strong>{check.name}</strong><span>{check.detail}</span></div></div>)}</div> : <span className="small-muted">{busy ? "Quality checks in progress…" : "Ready for quality checks"}</span>}</div></article>)}</div></>}
    <div className="privacy-note"><ShieldCheck size={20} /><div><strong>Your photos stay yours.</strong><p>Previews stay in your browser until you run checks. Uploaded images are processed in memory, not retained or used for training. Avoid showing faces, number plates, documents or private addresses.</p></div></div></div>
    <aside className="panel capture-guide"><span className="tiny-label">A BETTER CAPTURE</span><h3>Walk around.<br />Take your time.</h3><p>Use daylight and keep the whole vehicle in frame. These eight views are a useful starting point.</p><ol>{views.map((view, index) => <li key={view}><span>{String(index + 1).padStart(2, "0")}</span>{view}<Camera size={14} /></li>)}</ol><div className="guide-tip"><Info size={16} /><p>Blur and brightness thresholds are simple heuristics. A clear photo does not establish that a vehicle is undamaged.</p></div></aside></div>
  </div>;
}

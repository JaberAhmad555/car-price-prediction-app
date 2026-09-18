"use client";
export default function Error({ reset }: { reset: () => void }) { return <div className="container empty-page"><p className="eyebrow">A brief interruption</p><h1>Let’s try that again.</h1><p>The page couldn’t finish loading. Your images and vehicle details have not been saved.</p><button className="button" onClick={reset}>Reload this view</button></div>; }

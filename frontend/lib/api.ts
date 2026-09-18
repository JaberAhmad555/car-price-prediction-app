"use client";

import { useEffect, useState } from "react";

export const API_BASE = (process.env.NEXT_PUBLIC_API_URL || (process.env.NODE_ENV === "development" ? "http://127.0.0.1:8000" : "")).replace(/\/$/, "");
export type Metrics = { mae_bdt: number; rmse_bdt: number; r2: number };
export type Metadata = {
  model_ready: boolean; status_message: string | null; dataset_rows: number | null; raw_rows?: number;
  selected_model: string | null; test_metrics: Metrics | null; model_version?: string;
  dataset_description?: string; validation_metrics?: Record<string, Metrics>;
  split_rows?: Record<string, number>; range_method?: string;
};
export type Options = {
  available: boolean; brand: string[]; model: string[]; models_by_brand: Record<string, string[]>;
  transmission: string[]; fuel_type: string[]; body_type: string[];
  bounds: Record<string, { min: number; max: number }>;
};
export type Vehicle = {
  brand: string; model: string; year: number; mileage_km: number;
  transmission: string; fuel_type: string; body_type: string; engine_cc: number;
};
export type Quote = {
  estimated_price_bdt: number; estimated_price_lakh: number; lower_estimate: number; upper_estimate: number;
  range_method: string; model_version: string; dataset_description: string; disclaimer: string; vehicle: Vehicle;
};
export type Inspection = {
  checks: { name: string; passed: boolean; detail: string }[];
  notice: string; retention: string; width: number; height: number;
};
export type Insights = {
  available: boolean; data: null | {
    vehicle_count: number; average_price_bdt: number; median_price_bdt: number;
    top_brands: { name: string; count: number }[];
    price_by_year: { name: string; price_bdt: number; count: number }[];
    price_by_fuel: { name: string; price_bdt: number; count: number }[];
  };
};

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  if (!API_BASE) throw new Error("The valuation service is not configured. Please contact the site operator.");
  let response: Response;
  try {
    response = await fetch(API_BASE + path, { ...init, signal: init?.signal || AbortSignal.timeout(30000) });
  } catch {
    throw new Error("We couldn’t reach the service. Please try again in a moment.");
  }
  const body = await response.json();
  if (!response.ok) {
    throw new Error(typeof body.detail === "string" ? body.detail : "Please check your vehicle details and try again.");
  }
  return body as T;
}

export function useApi<T>(path: string) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 15000);
    request<T>(path, { signal: controller.signal })
      .then(value => { if (!controller.signal.aborted) { setData(value); setError(null); } })
      .catch((reason: Error) => { setError(reason.message); })
      .finally(() => { clearTimeout(timeout); setLoading(false); });
    return () => { controller.abort(); clearTimeout(timeout); };
  }, [path, attempt]);
  return { data, error, loading, retry: () => { setLoading(true); setAttempt(value => value + 1); } };
}

export function bdt(value: number, decimals = 0) { return "৳" + new Intl.NumberFormat("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals }).format(value); }
export function title(value: string) { return value.replace(/\b\w/g, letter => letter.toUpperCase()); }

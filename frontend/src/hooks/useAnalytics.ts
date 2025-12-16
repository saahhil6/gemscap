import { useState, useEffect, useCallback, useRef } from 'react';

export type AnalyticsParams = {
  symbol1?: string;
  symbol2?: string;
  timeframe?: string;
  windowSize?: number; // renamed to avoid colliding with global `window`
  regression?: 'ols' | 'huber' | 'theil_sen' | string;
  runAdf?: boolean;
};

export interface HedgeRatio {
  beta: number;
  alpha: number;
  r_squared: number;
  method?: string;
}

export interface CurrentValues {
  zscore: number;
  correlation: number;
  spread: number;
  asset1_price: number;
  asset2_price: number;
}

export interface TimeSeries {
  timestamps: string[];
  spread: number[];
  zscore: (number | null)[];
  correlation: (number | null)[];
  asset1_prices: number[];
  asset2_prices: number[];
}

export interface Signal {
  action: string;
  strength: number;
  description?: string;
  color?: string;
}

export interface AdfResult {
  statistic?: number;
  pvalue?: number;
  is_stationary?: boolean;
  error?: string;
}

export interface AnalyticsResult {
  hedge_ratio: HedgeRatio;
  current_values: CurrentValues;
  time_series: TimeSeries;
  adf_test: AdfResult | { skipped?: boolean };
  signal: Signal;
  data_points: number;
  window_size: number;
}

/**
 * useAnalytics hook
 * - fetches /api/analytics with the provided params
 * - returns { data, loading, error, refresh }
 * - cancels in-flight requests when params change
 * - optional polling via autoRefreshMs
 */
export default function useAnalytics(
  params: AnalyticsParams = {},
  autoRefreshMs?: number
): {
  data: AnalyticsResult | null;
  loading: boolean;
  error: string | null;
  refresh: () => void;
} {
  const {
    symbol1 = 'btcusdt',
    symbol2 = 'ethusdt',
    timeframe = '1m',
    windowSize = 20,
    regression = 'ols',
    runAdf = false
  } = params;

  const [data, setData] = useState<AnalyticsResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const controllerRef = useRef<AbortController | null>(null);

  const fetchAnalytics = useCallback(async () => {
    // Abort previous request if any
    if (controllerRef.current) controllerRef.current.abort();

    const controller = new AbortController();
    controllerRef.current = controller;

    setLoading(true);
    setError(null);

    try {
      const q = new URLSearchParams({
        symbol1,
        symbol2,
        timeframe,
        window: String(windowSize),
        regression,
        run_adf: String(runAdf)
      });

      const url = `/api/analytics?${q.toString()}`;

      const res = await fetch(url, { signal: controller.signal });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || res.statusText);
      }

      const json = await res.json();
      setData(json);
    } catch (err: any) {
      if (err?.name === 'AbortError') return;
      setError(err?.message || String(err));
      setData(null);
    } finally {
      setLoading(false);
    }
  }, [symbol1, symbol2, timeframe, windowSize, regression, runAdf]);

  // Fetch on param changes
  useEffect(() => {
    fetchAnalytics();
    return () => {
      controllerRef.current?.abort();
    };
  }, [fetchAnalytics]);

  // Optional polling
  useEffect(() => {
    if (!autoRefreshMs || autoRefreshMs <= 0) return;
    const id = setInterval(() => fetchAnalytics(), autoRefreshMs);
    return () => clearInterval(id);
  }, [fetchAnalytics, autoRefreshMs]);

  const refresh = useCallback(() => {
    fetchAnalytics();
  }, [fetchAnalytics]);

  return { data, loading, error, refresh };
}


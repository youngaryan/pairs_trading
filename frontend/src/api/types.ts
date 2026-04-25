export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };

export interface PaperTotals {
  equity: number;
  daily_pnl: number;
  rebalance_cost_pnl: number;
  cash: number;
  gross_exposure: number;
  gross_exposure_ratio: number;
  position_count: number;
  trade_count: number;
  turnover: number;
}

export interface LeaderboardRow {
  strategy: string;
  pipeline: string;
  mode: string;
  equity: number;
  return_since_inception: number;
  daily_pnl: number;
  trade_count: number;
  gross_exposure_ratio: number;
}

export interface PaperOrder {
  instrument?: string;
  side?: string;
  quantity?: number;
  mark_price?: number;
  execution_price?: number;
  target_weight?: number;
  commission?: number;
  notional?: number;
  [key: string]: unknown;
}

export interface PaperHistoryRow {
  timestamp: string;
  equity_after?: number;
  daily_pnl?: number;
  rebalance_cost_pnl?: number;
  net_return_since_inception?: number;
  cash_after?: number;
  gross_exposure_notional?: number;
  gross_exposure_ratio?: number;
  position_count?: number;
  trade_count?: number;
  turnover_notional?: number;
  [key: string]: unknown;
}

export interface PaperStrategy {
  name: string;
  pipeline: string;
  mode: string;
  equity: number;
  daily_pnl: number;
  rebalance_cost_pnl: number;
  return_since_inception: number;
  cash: number;
  gross_exposure: number;
  gross_exposure_ratio: number;
  position_count: number;
  trade_count: number;
  turnover: number;
  positions: Record<string, number>;
  target_weights: Record<string, number>;
  latest_orders: PaperOrder[];
  diagnostics: Record<string, unknown>;
  history: PaperHistoryRow[];
}

export interface PaperDashboardPayload {
  asof_date: string | null;
  run_timestamp_utc: string | null;
  totals: PaperTotals;
  leaderboard: LeaderboardRow[];
  strategies: PaperStrategy[];
  visuals: Record<string, unknown>;
}

export interface HealthResponse {
  status: string;
  service: string;
}

export interface StrategyCatalogItem {
  id: string;
  name: string;
  family: string;
  difficulty: string;
  pipeline: string;
  summary: string;
  how_it_works: string;
  best_for: string;
  watch_out: string;
  key_parameters: string[];
  example_cli: string;
  paper_config_example: Record<string, unknown>;
}

export interface BacktestTemplate {
  id: string;
  name: string;
  pipeline: string;
  symbols: string[];
  start: string;
  end: string;
  parameters: Record<string, unknown>;
  description: string;
  objective?: string;
  risk_level?: string;
  validation_focus?: string;
}

export interface BacktestRunRequest {
  pipeline: string;
  symbols: string[];
  start: string;
  end: string;
  interval: string;
  experiment_name?: string | null;
  sector_map_path?: string | null;
  event_file?: string | null;
  use_sec_companyfacts?: boolean;
  edgar_user_agent?: string | null;
  train_bars: number;
  test_bars: number;
  step_bars: number;
  bars_per_year: number;
  purge_bars: number;
  embargo_bars: number;
  pbo_partitions: number;
  parameters: Record<string, unknown>;
}

export interface BacktestJobResult {
  summary: Record<string, unknown>;
  validation: Record<string, unknown>;
  visuals: Record<string, unknown>;
  artifact_dir: string | null;
  fold_metrics_tail: JsonValue;
  equity_curve_tail: JsonValue;
  equity_curve_points: Array<{
    timestamp: string;
    equity: number;
    drawdown: number;
    net_return: number;
  }>;
  decision: {
    verdict: string;
    headline: string;
    passed_checks: number;
    total_checks: number;
    checks: Array<{
      name: string;
      value: number | null;
      passed: boolean;
      message: string;
    }>;
  };
}

export interface BacktestJob {
  id: string;
  status: "queued" | "running" | "completed" | "failed" | "interrupted" | string;
  request: BacktestRunRequest | Record<string, unknown>;
  created_at_utc: string;
  updated_at_utc: string;
  progress?: number;
  stage?: string;
  message?: string;
  warnings?: string[];
  started_at_utc?: string | null;
  finished_at_utc?: string | null;
  result?: BacktestJobResult | null;
  error?: string | null;
}

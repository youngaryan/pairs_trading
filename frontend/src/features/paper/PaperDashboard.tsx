import { useEffect, useState } from "react";
import {
  Activity,
  AlertCircle,
  BarChart3,
  BookOpen,
  FlaskConical,
  Database,
  GraduationCap,
  Layers3,
  Play,
  RefreshCw,
  Wallet
} from "lucide-react";

import { getBacktestTemplates, getHealth, getPaperSummary, getStrategyCatalog, runPaperBatch } from "../../api/client";
import type { BacktestTemplate, HealthResponse, PaperDashboardPayload, PaperOrder, PaperStrategy, StrategyCatalogItem } from "../../api/types";
import { MetricTile } from "../../components/MetricTile";
import { StatusBadge } from "../../components/StatusBadge";
import { Tabs, type TabItem } from "../../components/Tabs";
import { EquitySparkline } from "./EquitySparkline";
import { formatCurrency, formatNumber, formatPercent, toneFromValue } from "./format";
import { BacktestLab } from "./BacktestLab";
import { StrategyCatalog } from "./StrategyCatalog";
import { Tutorials } from "./Tutorials";

type DashboardView = "overview" | "strategies" | "orders" | "diagnostics" | "backtests" | "catalog" | "tutorials";

const tabs: TabItem<DashboardView>[] = [
  { id: "overview", label: "Overview", icon: <BarChart3 size={16} /> },
  { id: "strategies", label: "Strategies", icon: <Layers3 size={16} /> },
  { id: "orders", label: "Orders", icon: <Activity size={16} /> },
  { id: "diagnostics", label: "Diagnostics", icon: <Database size={16} /> },
  { id: "backtests", label: "Backtests", icon: <FlaskConical size={16} /> },
  { id: "catalog", label: "Catalog", icon: <BookOpen size={16} /> },
  { id: "tutorials", label: "Tutorials", icon: <GraduationCap size={16} /> }
];

function pipelineLabel(value: string) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function getAllOrders(strategies: PaperStrategy[]) {
  return strategies.flatMap((strategy) =>
    strategy.latest_orders.map((order) => ({
      strategy: strategy.name,
      ...order
    }))
  );
}

function StrategySelector({
  strategies,
  selectedName,
  onSelect
}: {
  strategies: PaperStrategy[];
  selectedName: string | null;
  onSelect: (strategyName: string) => void;
}) {
  return (
    <aside className="strategy-rail" aria-label="Strategies">
      {strategies.map((strategy) => (
        <button
          key={strategy.name}
          type="button"
          className={strategy.name === selectedName ? "strategy-rail__item strategy-rail__item--active" : "strategy-rail__item"}
          onClick={() => onSelect(strategy.name)}
        >
          <span>
            <strong>{strategy.name}</strong>
            <small>{pipelineLabel(strategy.pipeline)}</small>
          </span>
          <StatusBadge label={strategy.mode} tone={strategy.mode === "synthetic" ? "warning" : "neutral"} />
        </button>
      ))}
    </aside>
  );
}

function Leaderboard({ payload }: { payload: PaperDashboardPayload }) {
  return (
    <section className="panel">
      <div className="panel__header">
        <h2>Leaderboard</h2>
        <span>{payload.leaderboard.length} sleeves</span>
      </div>
      <div className="table-shell">
        <table>
          <thead>
            <tr>
              <th>Strategy</th>
              <th>Pipeline</th>
              <th>Equity</th>
              <th>Return</th>
              <th>Daily PnL</th>
              <th>Gross</th>
            </tr>
          </thead>
          <tbody>
            {payload.leaderboard.map((row) => (
              <tr key={row.strategy}>
                <td>
                  <strong>{row.strategy}</strong>
                  <small>{row.mode}</small>
                </td>
                <td>{pipelineLabel(row.pipeline)}</td>
                <td>{formatCurrency(row.equity)}</td>
                <td className={`numeric numeric--${toneFromValue(row.return_since_inception)}`}>
                  {formatPercent(row.return_since_inception)}
                </td>
                <td className={`numeric numeric--${toneFromValue(row.daily_pnl)}`}>{formatCurrency(row.daily_pnl)}</td>
                <td>{formatPercent(row.gross_exposure_ratio)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function WeightBars({ strategy }: { strategy: PaperStrategy }) {
  const rows = Object.entries(strategy.target_weights)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 12);

  if (!rows.length) {
    return <div className="empty-state">No target weights on the latest run.</div>;
  }

  return (
    <div className="weight-list">
      {rows.map(([instrument, weight]) => (
        <div key={instrument} className="weight-row">
          <span>{instrument}</span>
          <div className="weight-track">
            <i style={{ width: `${Math.min(Math.abs(weight) * 100, 100)}%` }} data-side={weight >= 0 ? "long" : "short"} />
          </div>
          <strong>{formatPercent(weight)}</strong>
        </div>
      ))}
    </div>
  );
}

function PositionTable({ strategy }: { strategy: PaperStrategy }) {
  const positions = Object.entries(strategy.positions).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));

  if (!positions.length) {
    return <div className="empty-state">No open positions.</div>;
  }

  return (
    <div className="table-shell table-shell--compact">
      <table>
        <thead>
          <tr>
            <th>Instrument</th>
            <th>Quantity</th>
            <th>Target</th>
          </tr>
        </thead>
        <tbody>
          {positions.map(([instrument, quantity]) => (
            <tr key={instrument}>
              <td>{instrument}</td>
              <td>{formatNumber(quantity)}</td>
              <td>{formatPercent(strategy.target_weights[instrument] ?? 0)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function OrdersTable({ orders }: { orders: Array<PaperOrder & { strategy?: string }> }) {
  if (!orders.length) {
    return <div className="empty-state">No simulated orders on the latest run.</div>;
  }

  return (
    <section className="panel">
      <div className="panel__header">
        <h2>Latest Orders</h2>
        <span>{orders.length} orders</span>
      </div>
      <div className="table-shell">
        <table>
          <thead>
            <tr>
              <th>Strategy</th>
              <th>Instrument</th>
              <th>Side</th>
              <th>Quantity</th>
              <th>Notional</th>
              <th>Commission</th>
              <th>Execution</th>
            </tr>
          </thead>
          <tbody>
            {orders.map((order, index) => (
              <tr key={`${order.strategy}-${order.instrument}-${index}`}>
                <td>{order.strategy ?? "-"}</td>
                <td>{order.instrument ?? "-"}</td>
                <td>
                  <StatusBadge
                    label={order.side ?? "-"}
                    tone={String(order.side).toLowerCase() === "buy" ? "positive" : "negative"}
                  />
                </td>
                <td>{formatNumber(Number(order.quantity ?? 0))}</td>
                <td>{formatCurrency(Number(order.notional ?? 0))}</td>
                <td>{formatCurrency(Number(order.commission ?? 0))}</td>
                <td>{formatCurrency(Number(order.execution_price ?? 0))}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function StrategyDetail({ strategy }: { strategy: PaperStrategy }) {
  return (
    <div className="strategy-detail">
      <section className="panel panel--chart">
        <div className="panel__header">
          <h2>{strategy.name}</h2>
          <StatusBadge label={pipelineLabel(strategy.pipeline)} />
        </div>
        <EquitySparkline history={strategy.history} />
      </section>
      <section className="panel">
        <div className="panel__header">
          <h2>Capital</h2>
          <span>{strategy.mode}</span>
        </div>
        <div className="mini-grid">
          <MetricTile label="Equity" value={formatCurrency(strategy.equity)} icon={<Wallet size={16} />} />
          <MetricTile
            label="Daily PnL"
            value={formatCurrency(strategy.daily_pnl)}
            tone={toneFromValue(strategy.daily_pnl)}
          />
          <MetricTile label="Return" value={formatPercent(strategy.return_since_inception)} tone={toneFromValue(strategy.return_since_inception)} />
          <MetricTile label="Gross Exposure" value={formatPercent(strategy.gross_exposure_ratio)} />
        </div>
      </section>
      <section className="panel">
        <div className="panel__header">
          <h2>Target Weights</h2>
          <span>{strategy.position_count} positions</span>
        </div>
        <WeightBars strategy={strategy} />
      </section>
      <section className="panel">
        <div className="panel__header">
          <h2>Positions</h2>
          <span>{formatCurrency(strategy.gross_exposure)}</span>
        </div>
        <PositionTable strategy={strategy} />
      </section>
    </div>
  );
}

export function PaperDashboard() {
  const [payload, setPayload] = useState<PaperDashboardPayload | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [catalog, setCatalog] = useState<StrategyCatalogItem[]>([]);
  const [backtestTemplates, setBacktestTemplates] = useState<BacktestTemplate[]>([]);
  const [activeTab, setActiveTab] = useState<DashboardView>("overview");
  const [selectedStrategyName, setSelectedStrategyName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRunning, setIsRunning] = useState(false);

  async function refresh() {
    setIsLoading(true);
    setError(null);
    try {
      const [healthResponse, paperResponse, catalogResponse, templateResponse] = await Promise.all([
        getHealth(),
        getPaperSummary(),
        getStrategyCatalog(),
        getBacktestTemplates()
      ]);
      setHealth(healthResponse);
      setPayload(paperResponse);
      setCatalog(catalogResponse);
      setBacktestTemplates(templateResponse);
      setSelectedStrategyName((current) => current ?? paperResponse.strategies[0]?.name ?? null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to load backend data.");
    } finally {
      setIsLoading(false);
    }
  }

  async function runBatch() {
    setIsRunning(true);
    setError(null);
    try {
      const nextPayload = await runPaperBatch();
      setPayload(nextPayload);
      setSelectedStrategyName((current) => current ?? nextPayload.strategies[0]?.name ?? null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Paper batch failed.");
    } finally {
      setIsRunning(false);
    }
  }

  useEffect(() => {
    void refresh();
  }, []);

  const selectedStrategy =
    payload?.strategies.find((strategy) => strategy.name === selectedStrategyName) ?? payload?.strategies[0] ?? null;
  const orders = payload ? getAllOrders(payload.strategies) : [];

  return (
    <main className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">Quant Operations</p>
          <h1>Paper trading console</h1>
          <span className="run-meta">
            {payload?.asof_date ?? "No as-of date"} | {payload?.run_timestamp_utc ?? "No run loaded"}
          </span>
        </div>
        <div className="header-actions">
          <StatusBadge label={health?.status === "ok" ? "Backend online" : "Backend unknown"} tone={health?.status === "ok" ? "positive" : "warning"} />
          <button type="button" className="icon-button" onClick={() => void refresh()} disabled={isLoading} title="Refresh data">
            <RefreshCw size={17} />
            <span>Refresh</span>
          </button>
          <button type="button" className="primary-button" onClick={() => void runBatch()} disabled={isRunning}>
            <Play size={17} />
            <span>{isRunning ? "Running" : "Run Batch"}</span>
          </button>
        </div>
      </header>

      {error ? (
        <section className="error-banner">
          <AlertCircle size={18} />
          <span>{error}</span>
        </section>
      ) : null}

      {payload ? (
        <>
          <section className="metrics-grid">
            <MetricTile label="Total Equity" value={formatCurrency(payload.totals.equity)} icon={<Wallet size={16} />} />
            <MetricTile
              label="Daily PnL"
              value={formatCurrency(payload.totals.daily_pnl)}
              tone={toneFromValue(payload.totals.daily_pnl)}
            />
            <MetricTile label="Cash" value={formatCurrency(payload.totals.cash)} />
            <MetricTile label="Gross Ratio" value={formatPercent(payload.totals.gross_exposure_ratio)} />
            <MetricTile label="Trades" value={formatNumber(payload.totals.trade_count)} />
            <MetricTile label="Turnover" value={formatCurrency(payload.totals.turnover)} />
          </section>

          <Tabs items={tabs} value={activeTab} onChange={setActiveTab} />

          {activeTab === "overview" ? (
            <div className="content-grid">
              <Leaderboard payload={payload} />
              {selectedStrategy ? <StrategyDetail strategy={selectedStrategy} /> : null}
            </div>
          ) : null}

          {activeTab === "strategies" ? (
            <div className="split-view">
              <StrategySelector
                strategies={payload.strategies}
                selectedName={selectedStrategy?.name ?? null}
                onSelect={setSelectedStrategyName}
              />
              {selectedStrategy ? <StrategyDetail strategy={selectedStrategy} /> : null}
            </div>
          ) : null}

          {activeTab === "orders" ? <OrdersTable orders={orders} /> : null}

          {activeTab === "diagnostics" ? (
            <section className="panel">
              <div className="panel__header">
                <h2>Diagnostics</h2>
                <span>{selectedStrategy?.name ?? "No strategy selected"}</span>
              </div>
              <pre className="json-panel">{JSON.stringify(selectedStrategy?.diagnostics ?? payload, null, 2)}</pre>
            </section>
          ) : null}

          {activeTab === "backtests" ? <BacktestLab catalog={catalog} templates={backtestTemplates} /> : null}

          {activeTab === "catalog" ? <StrategyCatalog catalog={catalog} /> : null}

          {activeTab === "tutorials" ? <Tutorials /> : null}
        </>
      ) : (
        <section className="loading-panel">{isLoading ? "Loading paper ledgers..." : "No paper data available."}</section>
      )}
    </main>
  );
}

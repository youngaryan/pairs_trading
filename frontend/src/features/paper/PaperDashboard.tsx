import { useEffect, useMemo, useState } from "react";
import { Activity, AlertCircle, BarChart3, BookOpen, Database, FlaskConical, GraduationCap, Layers3, RadioTower, RefreshCw } from "lucide-react";

import { getBacktestTemplates, getHealth, getPaperSummary, getStrategyCatalog } from "../../api/client";
import type { BacktestTemplate, HealthResponse, PaperDashboardPayload, StrategyCatalogItem } from "../../api/types";
import { StatusBadge } from "../../components/StatusBadge";
import { Tabs, type TabItem } from "../../components/Tabs";
import { BacktestLab } from "./BacktestLab";
import { LiveTradingPage } from "./LiveTradingPage";
import { OrdersPage } from "./OrdersPage";
import { OverviewPage } from "./OverviewPage";
import { StrategiesPage } from "./StrategiesPage";
import { StrategyCatalog } from "./StrategyCatalog";
import { Tutorials } from "./Tutorials";
import { getAllOrders } from "./paperUtils";

type DashboardView = "overview" | "live" | "strategies" | "orders" | "backtests" | "diagnostics" | "catalog" | "tutorials";

const tabs: TabItem<DashboardView>[] = [
  { id: "overview", label: "Overview", icon: <BarChart3 size={16} /> },
  { id: "live", label: "Live Trading", icon: <RadioTower size={16} /> },
  { id: "strategies", label: "Strategies", icon: <Layers3 size={16} /> },
  { id: "orders", label: "Orders", icon: <Activity size={16} /> },
  { id: "backtests", label: "Backtests", icon: <FlaskConical size={16} /> },
  { id: "diagnostics", label: "Diagnostics", icon: <Database size={16} /> },
  { id: "catalog", label: "Catalog", icon: <BookOpen size={16} /> },
  { id: "tutorials", label: "Tutorials", icon: <GraduationCap size={16} /> }
];

export function PaperDashboard() {
  const [payload, setPayload] = useState<PaperDashboardPayload | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [catalog, setCatalog] = useState<StrategyCatalogItem[]>([]);
  const [backtestTemplates, setBacktestTemplates] = useState<BacktestTemplate[]>([]);
  const [activeTab, setActiveTab] = useState<DashboardView>("overview");
  const [selectedStrategyName, setSelectedStrategyName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

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

  useEffect(() => {
    void refresh();
  }, []);

  const selectedStrategy =
    payload?.strategies.find((strategy) => strategy.name === selectedStrategyName) ?? payload?.strategies[0] ?? null;
  const orders = useMemo(() => (payload ? getAllOrders(payload.strategies) : []), [payload]);

  return (
    <main className="app-shell">
      <header className="app-header app-header--command">
        <div>
          <p className="eyebrow">Quant Operations</p>
          <h1>Trading operations console</h1>
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
          <Tabs items={tabs} value={activeTab} onChange={setActiveTab} />

          {activeTab === "overview" ? <OverviewPage payload={payload} selectedStrategy={selectedStrategy} /> : null}

          {activeTab === "live" ? (
            <LiveTradingPage payload={payload} catalog={catalog} onRunComplete={setPayload} onRefresh={() => void refresh()} />
          ) : null}

          {activeTab === "strategies" ? (
            <StrategiesPage strategies={payload.strategies} selectedStrategy={selectedStrategy} onSelect={setSelectedStrategyName} />
          ) : null}

          {activeTab === "orders" ? <OrdersPage orders={orders} /> : null}

          {activeTab === "backtests" ? <BacktestLab catalog={catalog} templates={backtestTemplates} /> : null}

          {activeTab === "diagnostics" ? (
            <section className="panel">
              <div className="panel__header">
                <h2>Diagnostics</h2>
                <span>{selectedStrategy?.name ?? "No strategy selected"}</span>
              </div>
              <pre className="json-panel">{JSON.stringify(selectedStrategy?.diagnostics ?? payload, null, 2)}</pre>
            </section>
          ) : null}

          {activeTab === "catalog" ? <StrategyCatalog catalog={catalog} /> : null}

          {activeTab === "tutorials" ? <Tutorials /> : null}
        </>
      ) : (
        <section className="loading-panel">{isLoading ? "Loading trading console..." : "No paper data available."}</section>
      )}
    </main>
  );
}

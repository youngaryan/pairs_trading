from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from ..config import BackendSettings
from ..schemas import PaperRunRequest
from ..services import PaperRunCommand, PaperService


def build_paper_router(settings: BackendSettings) -> APIRouter:
    router = APIRouter(prefix="/paper", tags=["paper"])
    service = PaperService(settings)

    @router.get("/summary")
    def get_summary(
        batch_summary_path: str | None = Query(default=None, description="Optional explicit paper_batch_summary.json path."),
    ) -> dict[str, Any]:
        return service.build_dashboard_payload(batch_summary_path=batch_summary_path)

    @router.get("/strategies")
    def list_strategies(
        batch_summary_path: str | None = Query(default=None, description="Optional explicit paper_batch_summary.json path."),
    ) -> list[dict[str, Any]]:
        return service.list_strategies(batch_summary_path=batch_summary_path)

    @router.get("/strategies/{strategy_name}")
    def get_strategy(
        strategy_name: str,
        batch_summary_path: str | None = Query(default=None, description="Optional explicit paper_batch_summary.json path."),
    ) -> dict[str, Any]:
        strategy = service.get_strategy(strategy_name=strategy_name, batch_summary_path=batch_summary_path)
        if strategy is None:
            raise HTTPException(status_code=404, detail=f"Strategy not found: {strategy_name}")
        return strategy

    @router.post("/run")
    def run_batch(request: PaperRunRequest) -> dict[str, Any]:
        try:
            return service.run_paper_batch(
                PaperRunCommand(
                    deployment_config_path=Path(request.deployment_config_path) if request.deployment_config_path else None,
                    asof_date=request.asof_date,
                )
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return router

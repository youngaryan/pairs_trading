from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from ...api import build_strategy_catalog


router = APIRouter(prefix="/strategies", tags=["strategies"])


@router.get("/catalog")
def get_strategy_catalog() -> list[dict[str, Any]]:
    return build_strategy_catalog()


@router.get("/catalog/{strategy_id}")
def get_strategy_catalog_item(strategy_id: str) -> dict[str, Any]:
    normalized = strategy_id.casefold()
    for item in build_strategy_catalog():
        if str(item["id"]).casefold() == normalized:
            return item
    raise HTTPException(status_code=404, detail=f"Strategy not found: {strategy_id}")

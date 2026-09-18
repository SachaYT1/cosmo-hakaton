"""Фабрика FastAPI-приложения: REST API + статический Leaflet-фронтенд."""

import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import healthz, router
from app.store import Catalog

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_CATALOG = Path(__file__).resolve().parent.parent / "data" / "catalog.gpkg"


def create_app(catalog_path: Path | None = None) -> FastAPI:
    path = Path(catalog_path or os.environ.get("CATALOG_PATH", DEFAULT_CATALOG))
    app = FastAPI(
        title="Мониторинг природных пожаров — информационно-аналитический сервис",
        version="0.1.0",
    )
    app.state.catalog = Catalog.load(path)
    app.include_router(router)
    app.add_api_route("/healthz", healthz, methods=["GET"])
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


app = create_app()

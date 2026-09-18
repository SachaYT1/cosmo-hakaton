"""FastAPI-приложение сервиса. Запуск: uvicorn fireapp.service.main:app --host 0.0.0.0 --port 8000"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from fireapp.service.routers import burns, hotspots, report

app = FastAPI(title="Мониторинг природных пожаров — API")

app.include_router(hotspots.router)
app.include_router(burns.router)
app.include_router(report.router)

static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

"""Подключение к PostGIS (SQLAlchemy) для сервиса."""
from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://fireapp:fireapp@db:5432/fireapp"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


# TODO: DDL / Alembic-миграции для таблиц:
#   hotspots(id, geom Point 4326, acq_datetime, chip_id)
#   burn_polygons(id, geom Polygon 4326, severity smallint, area_ha float, chip_id, date_pre, date_post)

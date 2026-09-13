"""Conexion compartida a MongoDB para todo el subsistema ml/.

Lee MONGODB_URI / MONGODB_DB (no las variables MONGO_HOST/PORT/DB del lado
Java). Carga .env si existe, porque los scripts de ml/ se ejecutan a mano.

MONGODB_DB no tiene valor por defecto a proposito: un default silencioso
apuntando a la base vieja (`F1-WeatherRec`) causo el incidente del
2026-09-11 (ver ARCHITECTURE.md). Es preferible fallar al arrancar.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database

_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / ".env")

logger = logging.getLogger(__name__)

_client: MongoClient | None = None


def get_db() -> Database:
    global _client
    db_name = os.getenv("MONGODB_DB")
    if not db_name:
        raise RuntimeError("MONGODB_DB no esta definida (ponla en .env, p.ej. F1-WeatherRec-Prod)")
    if _client is None:
        mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        _client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        _client.admin.command("ping")
        logger.info(f"Conectado a MongoDB en {mongo_uri} (db={db_name})")
    return _client[db_name]

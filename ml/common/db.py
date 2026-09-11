"""Conexion compartida a MongoDB para todo el subsistema ml/.

Reutiliza el mismo par de variables de entorno que python-app/server.py
(MONGODB_URI / MONGODB_DB), no las usadas en el lado Java (MONGO_HOST/PORT/DB).
Carga .env si existe (los scripts de ml/ se ejecutan manualmente, a diferencia
de server.py que ya recibe el entorno del proceso que lo lanza).
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
    if _client is None:
        mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        mongo_db_name = os.getenv("MONGODB_DB", "F1-WeatherRec")
        _client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        _client.admin.command("ping")
        logger.info(f"Conectado a MongoDB en {mongo_uri} (db={mongo_db_name})")
    return _client[os.getenv("MONGODB_DB", "F1-WeatherRec")]

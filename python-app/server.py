# server.py

import logging
import os
import sys
from pathlib import Path

from flask import Flask, jsonify
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# ml/ vive en la raiz del repo, no dentro de python-app/ -- se anade al
# path para poder montar ml.api.ml_blueprint sin duplicar codigo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.api.ml_blueprint import ml_bp  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.register_blueprint(ml_bp)

mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
mongo_db_name = os.getenv("MONGODB_DB", "F1-WeatherRec")

try:
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    logger.info(f"Conectado a MongoDB en {mongo_uri} (db={mongo_db_name})")
except PyMongoError as e:
    logger.error(f"No se pudo conectar a MongoDB en {mongo_uri}: {e}")
    sys.exit(1)

db = client[mongo_db_name]

positions_collection = db['positions']
sessions_collection = db['sessions']
drivers_collection = db['drivers']
weather_collection = db['weather']
stints_collection = db['stints']


@app.route('/consulta1', methods=['GET'])
def consulta1():
    logger.info("Ejecutando consulta1 (top 3 pilotos por circuito)")
    try:
        resultados = []
        session_keys = positions_collection.distinct("session_key")

        for session_key in session_keys:
            top_positions = positions_collection.find({"session_key": session_key}).sort("position", 1).limit(3)
            session_info = sessions_collection.find_one({"session_key": session_key}, {"circuit_short_name": 1, "_id": 0})

            circuito = session_info['circuit_short_name'] if session_info else f"session_{session_key}"

            resultado = {
                "circuito": circuito,
                "session_key": session_key,
                "top_positions": []
            }

            for position in top_positions:
                driver_info = drivers_collection.find_one(
                    {"driver_number": position['driver_number']},
                    {"full_name": 1, "_id": 0}
                )
                piloto = driver_info['full_name'] if driver_info else f"#{position['driver_number']}"
                resultado["top_positions"].append({
                    "posicion": position['position'],
                    "piloto": piloto
                })

            resultados.append(resultado)

        logger.info(f"consulta1 devolvio {len(resultados)} sesiones")
        return jsonify(resultados)
    except PyMongoError as e:
        logger.error(f"Error de MongoDB en consulta1: {e}")
        return jsonify({"error": "Error de base de datos"}), 500
    except Exception as e:
        logger.exception(f"Error inesperado en consulta1: {e}")
        return jsonify({"error": "Error interno"}), 500


@app.route('/consulta2', methods=['GET'])
def consulta2():
    logger.info("Ejecutando consulta2 (ganadores con lluvia + neumaticos)")
    try:
        resultados = []
        rainy_sessions = weather_collection.distinct("session_key", {"rainfall": 1})

        for session_key in rainy_sessions:
            session_info = sessions_collection.find_one({"session_key": session_key}, {"circuit_short_name": 1, "_id": 0})
            circuito = session_info['circuit_short_name'] if session_info else f"session_{session_key}"

            winner_position = positions_collection.find_one({"session_key": session_key, "position": 1}, {"driver_number": 1, "_id": 0})
            if not winner_position:
                continue

            driver_number = winner_position['driver_number']
            driver_info = drivers_collection.find_one({"driver_number": driver_number}, {"full_name": 1, "_id": 0})
            driver_name = driver_info['full_name'] if driver_info else f"#{driver_number}"

            stints = stints_collection.find({"session_key": session_key, "driver_number": driver_number}, {"compound": 1, "lap_start": 1, "lap_end": 1, "_id": 0})

            compound_usage = {}
            for stint in stints:
                compound = stint['compound']
                laps_used = stint['lap_end'] - stint['lap_start'] + 1
                compound_usage[compound] = compound_usage.get(compound, 0) + laps_used

            resultado = {
                "circuito": circuito,
                "piloto_ganador": driver_name,
                "compounds": compound_usage
            }

            resultados.append(resultado)

        logger.info(f"consulta2 devolvio {len(resultados)} sesiones con lluvia")
        return jsonify(resultados)
    except PyMongoError as e:
        logger.error(f"Error de MongoDB en consulta2: {e}")
        return jsonify({"error": "Error de base de datos"}), 500
    except Exception as e:
        logger.exception(f"Error inesperado en consulta2: {e}")
        return jsonify({"error": "Error interno"}), 500


@app.route('/consulta3', methods=['GET'])
def consulta3():
    logger.info("Ejecutando consulta3 (ganador + clima promedio)")
    try:
        resultados = []
        session_keys = positions_collection.distinct("session_key")

        for session_key in session_keys:
            winner_position = positions_collection.find_one({"session_key": session_key, "position": 1}, {"driver_number": 1, "_id": 0})
            if not winner_position:
                continue

            driver_number = winner_position['driver_number']
            driver_info = drivers_collection.find_one({"driver_number": driver_number}, {"full_name": 1, "_id": 0})
            driver_name = driver_info['full_name'] if driver_info else f"#{driver_number}"

            compound_usage = {}
            stints = stints_collection.find({"session_key": session_key, "driver_number": driver_number}, {"compound": 1, "lap_start": 1, "lap_end": 1, "_id": 0})
            for stint in stints:
                compound = stint['compound']
                laps_used = stint['lap_end'] - stint['lap_start'] + 1
                compound_usage[compound] = compound_usage.get(compound, 0) + laps_used

            weather_conditions = weather_collection.find({"session_key": session_key}, {"_id": 0, "rainfall": 1, "track_temperature": 1, "air_temperature": 1, "humidity": 1, "wind_speed": 1})

            total_rainfall = 0
            total_track_temp = 0
            total_air_temp = 0
            total_humidity = 0
            total_wind_speed = 0
            count = 0

            for condition in weather_conditions:
                total_rainfall += condition.get('rainfall', 0)
                total_track_temp += condition.get('track_temperature', 0)
                total_air_temp += condition.get('air_temperature', 0)
                total_humidity += condition.get('humidity', 0)
                total_wind_speed += condition.get('wind_speed', 0)
                count += 1

            if count > 0:
                avg_track_temp = total_track_temp / count
                avg_air_temp = total_air_temp / count
                avg_humidity = total_humidity / count
                avg_wind_speed = total_wind_speed / count
                rainfall_binary = 1 if total_rainfall > 0 else 0
            else:
                avg_track_temp = None
                avg_air_temp = None
                avg_humidity = None
                avg_wind_speed = None
                rainfall_binary = 0

            session_info = sessions_collection.find_one({"session_key": session_key}, {"circuit_short_name": 1, "country_name": 1, "date_start": 1, "_id": 0})

            resultado = {
                "circuito": session_info['circuit_short_name'] if session_info else f"session_{session_key}",
                "pais": session_info['country_name'] if session_info else None,
                "fecha_inicio": session_info['date_start'] if session_info else None,
                "piloto_ganador": driver_name,
                "compounds": compound_usage,
                "weather_conditions": {
                    "lluvia": rainfall_binary,
                    "temp_pista_promedio": avg_track_temp,
                    "temp_aire_promedio": avg_air_temp,
                    "humedad_promedio": avg_humidity,
                    "viento_promedio": avg_wind_speed
                }
            }

            resultados.append(resultado)

        logger.info(f"consulta3 devolvio {len(resultados)} sesiones")
        return jsonify(resultados)
    except PyMongoError as e:
        logger.error(f"Error de MongoDB en consulta3: {e}")
        return jsonify({"error": "Error de base de datos"}), 500
    except Exception as e:
        logger.exception(f"Error inesperado en consulta3: {e}")
        return jsonify({"error": "Error interno"}), 500


if __name__ == '__main__':
    app.run(debug=True, host=os.getenv("FLASK_HOST", "127.0.0.1"), port=int(os.getenv("FLASK_PORT", "5000")))

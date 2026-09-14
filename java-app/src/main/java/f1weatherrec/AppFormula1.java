package f1weatherrec;

import org.json.JSONArray;

import java.time.Year;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

public class AppFormula1 {
    private static final Logger LOGGER = Logger.getLogger(AppFormula1.class.getName());

    public static void main(String[] args) {
        LogConfig.configure();
        boolean incremental = args.length > 0 && args[0].equals("incremental");
        LOGGER.info("Iniciando ingesta de datos de Formula 1" + (incremental ? " (incremental)" : "") + "...");

        List<Integer> years = Arrays.asList(2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026);
        List<Integer> modernYears = Arrays.asList(2023, 2024, 2025, 2026);
        if (incremental) {
            // Solo el anio en curso: lo unico que puede haber cambiado desde la ultima vez.
            years = List.of(Year.now().getValue());
            modernYears = years;
        }

        try (MongoDBClient mongoDBClient = new MongoDBClient()) {
            // Race para todo el rango (OpenF1 solo tiene datos reales desde 2023,
            // los anios anteriores devuelven 404 y se registran como advertencia).
            // Qualifying/Sprint Qualifying/Sprint solo se piden para 2023+: son las
            // sesiones que sustentan el modelo "era moderna" (decision: precision
            // maxima 2023-2026, ver ml/docs/ARCHITECTURE.md), y no existen antes.
            JSONArray sessionsArray = new JSONArray(Formula1Service.getSessions(years, "Race"));
            appendSessions(sessionsArray, Formula1Service.getSessions(modernYears, "Qualifying"));
            appendSessions(sessionsArray, Formula1Service.getSessions(modernYears, "Sprint Qualifying"));
            appendSessions(sessionsArray, Formula1Service.getSessions(modernYears, "Sprint"));

            if (incremental) {
                // Una sesion con vueltas guardadas ya esta completa: no se vuelve a pedir.
                Set<Integer> done = mongoDBClient.sessionKeysWithLaps();
                JSONArray pending = new JSONArray();
                for (int i = 0; i < sessionsArray.length(); i++) {
                    if (!done.contains(sessionsArray.getJSONObject(i).getInt("session_key"))) {
                        pending.put(sessionsArray.getJSONObject(i));
                    }
                }
                LOGGER.info("Sesiones nuevas: " + pending.length() + " de " + sessionsArray.length());
                sessionsArray = pending;
            }

            if (sessionsArray.length() == 0) {
                LOGGER.info("No hay sesiones que ingerir para " + years + ".");
                return;
            }
            LOGGER.info("Total de sesiones a ingerir (Race+Qualifying+Sprint Qualifying+Sprint): " + sessionsArray.length());

            saveSessions(mongoDBClient, sessionsArray);
            saveDrivers(mongoDBClient, sessionsArray);
            saveStints(mongoDBClient, sessionsArray);
            saveLaps(mongoDBClient, sessionsArray);
            savePit(mongoDBClient, sessionsArray);
            saveWeather(mongoDBClient, sessionsArray);
            savePositions(mongoDBClient, sessionsArray);
        } catch (Exception e) {
            LOGGER.severe("No se pudo conectar a MongoDB, se aborta la ingesta: " + e.getMessage());
            return;
        }

        LOGGER.info("Ingesta de datos de Formula 1 finalizada.");
    }

    private static void appendSessions(JSONArray target, String rawJson) {
        JSONArray source = new JSONArray(rawJson);
        for (int i = 0; i < source.length(); i++) {
            target.put(source.getJSONObject(i));
        }
    }

    private static void saveSessions(MongoDBClient client, JSONArray sessionsArray) {
        try {
            JSONArray processed = FormulaDataPreprocessor.processSessionsData(sessionsArray);
            client.insertSessionsData(processed);
            LOGGER.info("Sesiones guardadas: " + processed.length());
        } catch (Exception e) {
            LOGGER.severe("Fallo guardando sesiones: " + e.getMessage());
        }
    }

    private static void saveDrivers(MongoDBClient client, JSONArray sessionsArray) {
        try {
            String drivers = Formula1Service.getDrivers(sessionsArray);
            JSONArray processed = FormulaDataPreprocessor.processDriversData(new JSONArray(drivers));
            client.insertDriversData(processed);
            LOGGER.info("Pilotos guardados: " + processed.length());
        } catch (Exception e) {
            LOGGER.severe("Fallo guardando pilotos: " + e.getMessage());
        }
    }

    private static void saveStints(MongoDBClient client, JSONArray sessionsArray) {
        try {
            String stints = Formula1Service.getStints(sessionsArray);
            JSONArray processed = FormulaDataPreprocessor.processStintsData(new JSONArray(stints));
            client.insertStintsData(processed);
            LOGGER.info("Stints guardados: " + processed.length());
        } catch (Exception e) {
            LOGGER.severe("Fallo guardando stints: " + e.getMessage());
        }
    }

    private static void saveLaps(MongoDBClient client, JSONArray sessionsArray) {
        try {
            String laps = Formula1Service.getLaps(sessionsArray);
            JSONArray processed = FormulaDataPreprocessor.processLapsData(new JSONArray(laps));
            client.insertLapsData(processed);
            LOGGER.info("Vueltas guardadas: " + processed.length());
        } catch (Exception e) {
            LOGGER.severe("Fallo guardando vueltas: " + e.getMessage());
        }
    }

    private static void savePit(MongoDBClient client, JSONArray sessionsArray) {
        try {
            String pit = Formula1Service.getPit(sessionsArray);
            JSONArray processed = FormulaDataPreprocessor.processPitData(new JSONArray(pit));
            client.insertPitData(processed);
            LOGGER.info("Paradas en boxes guardadas: " + processed.length());
        } catch (Exception e) {
            LOGGER.severe("Fallo guardando paradas en boxes: " + e.getMessage());
        }
    }

    private static void saveWeather(MongoDBClient client, JSONArray sessionsArray) {
        try {
            String weather = Formula1Service.getWeather(sessionsArray);
            JSONArray processed = FormulaDataPreprocessor.processWeatherData(new JSONArray(weather));
            client.insertWeatherData(processed);
            LOGGER.info("Registros de clima guardados: " + processed.length());
        } catch (Exception e) {
            LOGGER.severe("Fallo guardando clima: " + e.getMessage());
        }
    }

    private static void savePositions(MongoDBClient client, JSONArray sessionsArray) {
        try {
            String positions = Formula1Service.getPosition(sessionsArray);
            JSONArray processed = FormulaDataPreprocessor.processPositionData(new JSONArray(positions));
            client.insertPositionsData(processed);
            LOGGER.info("Posiciones guardadas: " + processed.length());
        } catch (Exception e) {
            LOGGER.severe("Fallo guardando posiciones: " + e.getMessage());
        }
    }
}

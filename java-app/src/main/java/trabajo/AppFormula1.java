package trabajo;

import org.json.JSONArray;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

public class AppFormula1 {
    private static final Logger LOGGER = Logger.getLogger(AppFormula1.class.getName());

    public static void main(String[] args) {
        LogConfig.configure();
        LOGGER.info("Iniciando ingesta de datos de Formula 1...");

        List<Integer> years = Arrays.asList(2023, 2024);

        JSONArray sessionsArray;
        try {
            String sessions = Formula1Service.getSessions(years);
            sessionsArray = new JSONArray(sessions);
        } catch (IOException e) {
            LOGGER.severe("No se pudieron obtener las sesiones, se aborta la ingesta: " + e.getMessage());
            return;
        }

        try (MongoDBClient mongoDBClient = new MongoDBClient()) {
            saveSessions(mongoDBClient, sessionsArray);
            saveDrivers(mongoDBClient, sessionsArray);
            saveStints(mongoDBClient, sessionsArray);
            saveWeather(mongoDBClient, sessionsArray);
            savePositions(mongoDBClient, sessionsArray);
        } catch (Exception e) {
            LOGGER.severe("No se pudo conectar a MongoDB, se aborta la ingesta: " + e.getMessage());
            return;
        }

        LOGGER.info("Ingesta de datos de Formula 1 finalizada.");
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

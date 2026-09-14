package trabajo;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.HashMap;
import java.util.Map;

public class FormulaDataPreprocessor {

    public static JSONArray processSessionsData(JSONArray rawSessions) {
        JSONArray processedSessions = new JSONArray();
        for (int i = 0; i < rawSessions.length(); i++) {
            JSONObject rawSession = rawSessions.getJSONObject(i);
            JSONObject session = new JSONObject();
            session.put("session_key", rawSession.getInt("session_key"));
            session.put("session_name", rawSession.getString("session_name"));
            session.put("country_name", rawSession.getString("country_name"));
            session.put("country_code", rawSession.getString("country_code"));
            session.put("circuit_short_name", rawSession.getString("circuit_short_name"));
            session.put("year", rawSession.getInt("year"));
            session.put("date_start", rawSession.getString("date_start"));
            session.put("date_end", rawSession.getString("date_end"));
            processedSessions.put(session);
        }
        return processedSessions;
    }

    public static JSONArray processDriversData(JSONArray arrayDrivers) {
        JSONArray processedDrivers = new JSONArray();
        for (int i = 0; i < arrayDrivers.length(); i++) {
            JSONObject rawDriver = arrayDrivers.getJSONObject(i);
            JSONObject driver = new JSONObject();
            driver.put("driver_number", rawDriver.getInt("driver_number"));
            driver.put("full_name", rawDriver.getString("full_name"));
            driver.put("session_key", rawDriver.getInt("session_key"));
            processedDrivers.put(driver);
        }
        return processedDrivers;
    }

    public static JSONArray processStintsData(JSONArray arrayStints) {
        for (int i = 0; i < arrayStints.length(); i++) {
            JSONObject rawStint = arrayStints.getJSONObject(i);
            rawStint.remove("meeting_key");
        }
        return arrayStints;
    }

    public static JSONArray processLapsData(JSONArray arrayLaps) {
        for (int i = 0; i < arrayLaps.length(); i++) {
            JSONObject rawLap = arrayLaps.getJSONObject(i);
            rawLap.remove("meeting_key");
        }
        return arrayLaps;
    }

    public static JSONArray processPitData(JSONArray arrayPit) {
        for (int i = 0; i < arrayPit.length(); i++) {
            JSONObject rawPit = arrayPit.getJSONObject(i);
            rawPit.remove("meeting_key");
        }
        return arrayPit;
    }

    public static JSONArray processWeatherData(JSONArray arrayWeather) {
        for (int i = 0; i < arrayWeather.length(); i++) {
            JSONObject rawWeatherData = arrayWeather.getJSONObject(i);
            rawWeatherData.remove("meeting_key");
            rawWeatherData.remove("wind_direction");
            rawWeatherData.remove("pressure");
        }
        return arrayWeather;
    }

    public static JSONArray processPositionData(JSONArray arrayPositions) {
        Map<String, JSONObject> finalPositions = new HashMap<>();
        for (int i = 0; i < arrayPositions.length(); i++) {
            JSONObject rawPosition = arrayPositions.getJSONObject(i);
            int driver_number = rawPosition.getInt("driver_number");
            int session_key = rawPosition.getInt("session_key");
            String key = driver_number + "-" + session_key;
            finalPositions.put(key, rawPosition);
        }
        JSONArray processedPositions = new JSONArray();
        for (JSONObject position : finalPositions.values()) {
            processedPositions.put(position);
        }
        return processedPositions;
    }

}
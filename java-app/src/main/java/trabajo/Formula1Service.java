package trabajo;

import org.apache.http.HttpResponse;
import org.apache.http.client.fluent.Request;
import org.apache.http.client.fluent.Response;
import org.apache.http.util.EntityUtils;
import org.json.JSONArray;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.URLEncoder;
import java.util.List;
import java.util.logging.Logger;

public class Formula1Service {
    private static final Logger LOGGER = Logger.getLogger(Formula1Service.class.getName());
    private static final String URL_TEMPLATE = "https://api.openf1.org/v1/";
    private static final int MAX_RETRIES = 8;

    public static String getSessions(List<Integer> years) {
        return getSessions(years, "Race");
    }

    public static String getSessions(List<Integer> years, String sessionName) {
        LOGGER.info("Solicitando sesiones '" + sessionName + "' para los anios: " + years);
        JSONArray sessions = new JSONArray();
        String encodedName;
        try {
            encodedName = URLEncoder.encode(sessionName, "UTF-8");
        } catch (UnsupportedEncodingException e) {
            encodedName = sessionName;
        }
        for (int year : years) {
            try {
                String response = makeRequest("sessions?year=" + year + "&session_name=" + encodedName);
                JSONArray sessionsArray = new JSONArray(response);
                for (int i = 0; i < sessionsArray.length(); i++) {
                    sessions.put(sessionsArray.getJSONObject(i));
                }
            } catch (Exception e) {
                LOGGER.warning("Sin datos de sesiones '" + sessionName + "' para year=" + year + ": " + e.getMessage());
            }
        }
        LOGGER.info("Sesiones '" + sessionName + "' obtenidas: " + sessions.length());
        return sessions.toString();
    }
    public static String getDrivers(JSONArray sessionArray) {
        JSONArray drivers = new JSONArray();
        for (int i = 0; i < sessionArray.length(); i++) {
            int session_key = sessionArray.getJSONObject(i).getInt("session_key");
            try {
                String response = makeRequest("drivers?session_key=" + session_key);
                JSONArray driversArray = new JSONArray(response);
                for (int j = 0; j < driversArray.length(); j++) {
                    drivers.put(driversArray.getJSONObject(j));
                }
            } catch (Exception e) {
                LOGGER.warning("Sin datos de pilotos para session_key=" + session_key + ": " + e.getMessage());
            }
        }
        return drivers.toString();
    }

    public static String getStints(JSONArray sessionsArray) {
        JSONArray stints = new JSONArray();
        for (int i = 0; i < sessionsArray.length(); i++) {
            int session_key = sessionsArray.getJSONObject(i).getInt("session_key");
            try {
                String response = makeRequest("stints?session_key=" + session_key);
                JSONArray stintsArray = new JSONArray(response);
                for (int j = 0; j < stintsArray.length(); j++) {
                    stints.put(stintsArray.getJSONObject(j));
                }
            } catch (Exception e) {
                LOGGER.warning("Sin datos de stints para session_key=" + session_key + ": " + e.getMessage());
            }
        }
        return stints.toString();
    }

    public static String getPosition(JSONArray sessionsArray) {
        JSONArray positions = new JSONArray();
        for (int i = 0; i < sessionsArray.length(); i++) {
            int session_key = sessionsArray.getJSONObject(i).getInt("session_key");
            try {
                String response = makeRequest("position?session_key=" + session_key);
                JSONArray positionsArray = new JSONArray(response);
                for (int j = 0; j < positionsArray.length(); j++) {
                    positions.put(positionsArray.getJSONObject(j));
                }
            } catch (Exception e) {
                LOGGER.warning("Sin datos de posiciones para session_key=" + session_key + ": " + e.getMessage());
            }
        }
        return positions.toString();
    }

    public static String getWeather(JSONArray sessionsArray) {
        JSONArray weather = new JSONArray();
        for (int i = 0; i < sessionsArray.length(); i++) {
            int session_key = sessionsArray.getJSONObject(i).getInt("session_key");
            try {
                String response = makeRequest("weather?session_key=" + session_key);
                JSONArray weatherArray = new JSONArray(response);
                for (int j = 0; j < weatherArray.length(); j++) {
                    weather.put(weatherArray.getJSONObject(j));
                }
            } catch (Exception e) {
                LOGGER.warning("Sin datos de clima para session_key=" + session_key + ": " + e.getMessage());
            }
        }
        return weather.toString();
    }

    public static String getLaps(JSONArray sessionsArray) {
        JSONArray laps = new JSONArray();
        for (int i = 0; i < sessionsArray.length(); i++) {
            int session_key = sessionsArray.getJSONObject(i).getInt("session_key");
            try {
                String response = makeRequest("laps?session_key=" + session_key);
                JSONArray lapsArray = new JSONArray(response);
                for (int j = 0; j < lapsArray.length(); j++) {
                    laps.put(lapsArray.getJSONObject(j));
                }
            } catch (Exception e) {
                LOGGER.warning("Sin datos de vueltas para session_key=" + session_key + ": " + e.getMessage());
            }
        }
        return laps.toString();
    }

    public static String getPit(JSONArray sessionsArray) {
        JSONArray pits = new JSONArray();
        for (int i = 0; i < sessionsArray.length(); i++) {
            int session_key = sessionsArray.getJSONObject(i).getInt("session_key");
            try {
                String response = makeRequest("pit?session_key=" + session_key);
                JSONArray pitArray = new JSONArray(response);
                for (int j = 0; j < pitArray.length(); j++) {
                    pits.put(pitArray.getJSONObject(j));
                }
            } catch (Exception e) {
                LOGGER.warning("Sin datos de paradas en boxes para session_key=" + session_key + ": " + e.getMessage());
            }
        }
        return pits.toString();
    }

    private static String makeRequest(String request) throws IOException {
        return makeRequest(request, 1);
    }

    private static String makeRequest(String request, int attempt) throws IOException {
        String url = URL_TEMPLATE + request;
        Response response = Request.Get(url).execute();
        HttpResponse httpResponse = response.returnResponse();
        String responseString = EntityUtils.toString(httpResponse.getEntity());
        int statusCode = httpResponse.getStatusLine().getStatusCode();

        if (statusCode == 429) {
            if (attempt >= MAX_RETRIES) {
                throw new IOException("OpenF1 sigue devolviendo 429 (rate limit) tras " + MAX_RETRIES
                        + " intentos para: " + request);
            }
            long waitMs = Math.min(1000L * (1L << attempt), 20000L); // exponencial, techo 20s
            LOGGER.warning("Rate limit (429) en '" + request + "', reintento " + attempt + "/" + MAX_RETRIES
                    + " en " + (waitMs / 1000.0) + "s");
            try {
                Thread.sleep(waitMs);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("Interrumpido esperando reintento de: " + request, e);
            }
            return makeRequest(request, attempt + 1);
        }

        if (statusCode >= 400) {
            LOGGER.severe("OpenF1 respondio " + statusCode + " para '" + request + "': " + responseString);
            throw new IOException("OpenF1 devolvio codigo " + statusCode + " para " + request);
        }

        return responseString;
    }

}

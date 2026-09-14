// WeatherService.java
package f1weatherrec;

import org.apache.http.client.fluent.Request;
import org.apache.http.client.fluent.Response;

import java.io.IOException;
import java.util.logging.Logger;

public class WeatherService {
    private static final Logger LOGGER = Logger.getLogger(WeatherService.class.getName());
    private static final String URL_TEMPLATE =
            "https://api.openweathermap.org/data/2.5/forecast?q=%s&APPID=%s&units=metric";
    private static final String URL_TEMPLATE_COORDS =
            "https://api.openweathermap.org/data/2.5/forecast?lat=%s&lon=%s&APPID=%s&units=metric";

    private static String apiKey() {
        String apiKey = System.getenv("OPENWEATHER_API_KEY");
        if (apiKey == null || apiKey.isEmpty()) {
            LOGGER.severe("OPENWEATHER_API_KEY no esta configurada en el entorno.");
            throw new IllegalStateException("OPENWEATHER_API_KEY env var is not set.");
        }
        return apiKey;
    }

    public static String getWeatherData(String city, String countryCode) throws IOException {
        String url = String.format(URL_TEMPLATE, city + "," + countryCode, apiKey());
        try {
            Response response = Request.Get(url).execute();
            return response.returnContent().asString();
        } catch (IOException e) {
            LOGGER.severe("Fallo al obtener el pronostico para " + city + "," + countryCode + ": " + e.getMessage());
            throw e;
        }
    }

    /**
     * Pronostico por coordenadas del CIRCUITO (no de la ciudad): evita el
     * desajuste ciudad-circuito (Sakhir esta a ~30km de Manama) y elimina la
     * necesidad de mantener a mano una lista de ciudades -- las coordenadas
     * salen de circuits.json, que ya es la fuente unica de circuitos.
     */
    public static String getWeatherDataByCoords(double lat, double lon) throws IOException {
        String url = String.format(java.util.Locale.US, URL_TEMPLATE_COORDS, lat, lon, apiKey());
        try {
            Response response = Request.Get(url).execute();
            return response.returnContent().asString();
        } catch (IOException e) {
            LOGGER.severe("Fallo al obtener el pronostico para lat=" + lat + " lon=" + lon + ": " + e.getMessage());
            throw e;
        }
    }
}

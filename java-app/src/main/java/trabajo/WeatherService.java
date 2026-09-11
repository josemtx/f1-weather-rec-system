// WeatherService.java
package trabajo;

import org.apache.http.client.fluent.Request;
import org.apache.http.client.fluent.Response;

import java.io.IOException;
import java.util.logging.Logger;

public class WeatherService {
    private static final Logger LOGGER = Logger.getLogger(WeatherService.class.getName());
    private static final String URL_TEMPLATE =
            "https://api.openweathermap.org/data/2.5/forecast?q=%s&APPID=%s&units=metric";

    public static String getWeatherData(String city, String countryCode) throws IOException {
        String apiKey = System.getenv("OPENWEATHER_API_KEY");
        if (apiKey == null || apiKey.isEmpty()) {
            LOGGER.severe("OPENWEATHER_API_KEY no esta configurada en el entorno.");
            throw new IllegalStateException("OPENWEATHER_API_KEY env var is not set.");
        }
        String url = String.format(URL_TEMPLATE, city + "," + countryCode, apiKey);
        try {
            Response response = Request.Get(url).execute();
            return response.returnContent().asString();
        } catch (IOException e) {
            LOGGER.severe("Fallo al obtener el pronostico para " + city + "," + countryCode + ": " + e.getMessage());
            throw e;
        }
    }
}

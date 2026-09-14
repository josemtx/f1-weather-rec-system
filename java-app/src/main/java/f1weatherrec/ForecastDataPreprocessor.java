package f1weatherrec;

import org.json.JSONArray;
import org.json.JSONObject;

import java.time.Instant;

/**
 * Convierte la respuesta del pronostico de OpenWeatherMap al esquema que
 * consume el modelo.
 *
 * Los nombres de campo se alinean deliberadamente con los de
 * `climate_historical` (temp_2m_avg, humidity_relative, wind_speed_10m,
 * precipitation_mm...) para que el feature engineering pueda tratar el
 * pronostico como una fuente mas de clima sin traducir nombres.
 *
 * Dos decisiones respecto a la version anterior:
 *  - Se guardan las 40 entradas de 3h, no solo la de las 12:00 UTC: las
 *    carreras se disputan a horas muy distintas y la de mediodia rara vez
 *    coincide con la hora de carrera.
 *  - Se guarda `forecast_fetched_at`: sin saber con cuanta antelacion se
 *    emitio un pronostico no se puede corregir su sesgo despues contra lo
 *    que realmente ocurrio (ver ml/docs/ARCHITECTURE.md, enfoque B).
 */
public class ForecastDataPreprocessor {

    public JSONArray processForecastData(String rawJson, String circuitShortName) {
        JSONObject json = new JSONObject(rawJson);
        JSONArray list = json.getJSONArray("list");
        String fetchedAt = Instant.now().toString();

        JSONArray processedList = new JSONArray();

        for (int i = 0; i < list.length(); i++) {
            JSONObject forecast = list.getJSONObject(i);
            JSONObject main = forecast.getJSONObject("main");
            JSONObject wind = forecast.getJSONObject("wind");
            JSONObject weather = forecast.getJSONArray("weather").getJSONObject(0);
            JSONObject clouds = forecast.optJSONObject("clouds");

            // `rain` solo existe en la respuesta cuando se espera lluvia:
            // su ausencia significa 0 mm, no un dato que falte.
            JSONObject rain = forecast.optJSONObject("rain");
            double precipitationMm = rain != null ? rain.optDouble("3h", 0.0) : 0.0;

            JSONObject out = new JSONObject()
                    .put("circuit_short_name", circuitShortName)
                    .put("datetime", forecast.getString("dt_txt"))
                    .put("forecast_fetched_at", fetchedAt)
                    .put("description", weather.getString("description"))
                    .put("temp_2m_avg", main.getDouble("temp"))
                    .put("temp_2m_max", main.getDouble("temp_max"))
                    .put("temp_2m_min", main.getDouble("temp_min"))
                    .put("humidity_relative", main.getDouble("humidity"))
                    .put("surface_pressure", main.getDouble("pressure"))
                    .put("wind_speed_10m", wind.getDouble("speed"))
                    .put("precipitation_mm", precipitationMm)
                    // `pop` es la probabilidad de precipitacion (0-1) segun el
                    // propio modelo meteorologico: es la incertidumbre que el
                    // simulador Monte Carlo necesita para muestrear lluvia.
                    .put("precipitation_probability", forecast.optDouble("pop", 0.0))
                    .put("cloud_cover", clouds != null ? clouds.optDouble("all", Double.NaN) : Double.NaN)
                    .put("dew_point", main.optDouble("dew_point", Double.NaN));

            processedList.put(out);
        }

        return processedList;
    }
}

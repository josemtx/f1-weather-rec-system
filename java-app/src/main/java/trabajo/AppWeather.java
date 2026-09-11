package trabajo;

import com.mongodb.MongoClient;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoDatabase;
import com.mongodb.client.model.Filters;
import com.mongodb.client.model.IndexOptions;
import com.mongodb.client.model.Indexes;
import com.mongodb.client.model.ReplaceOptions;
import org.bson.Document;
import org.bson.conversions.Bson;
import org.json.JSONArray;

import java.util.logging.Logger;

public class AppWeather {
    private static final Logger LOGGER = Logger.getLogger(AppWeather.class.getName());

    public static void main(String[] args) {
        LogConfig.configure();
        LOGGER.info("Iniciando ingesta de pronosticos de clima...");

        String[] cities = {
                "Manama,bh", "Jeddah,sa", "Melbourne,au", "Suzuka,jp", "Shanghai,cn",
                "Miami,us", "Imola,it", "Monte-Carlo,mc", "Montreal,ca", "Barcelona,es",
                "Spielberg,at", "Silverstone,uk", "Budapest,hu", "Spa,be", "Zandvoort,nl",
                "Monza,it", "Baku,az", "Singapore,sg", "Austin,us", "Ciudad%20de%20Mexico,mx",
                "Sao%20Paulo,br", "Las%20Vegas,us", "Lusail,qa", "Abu%20Dhabi,ae"
        };

        int succeeded = 0;
        int failed = 0;

        try (MongoClient mongoClient = new MongoClient("localhost", 27017)) {
            MongoDatabase database = mongoClient.getDatabase("F1-WeatherRec");
            MongoCollection<Document> collection = database.getCollection("forecast_data");
            collection.createIndex(Indexes.ascending("city", "datetime"), new IndexOptions().unique(true));

            ForecastDataPreprocessor preprocessor = new ForecastDataPreprocessor();

            for (String city : cities) {
                String cityName = city.split(",")[0];
                try {
                    String rawJson = WeatherService.getWeatherData(cityName, city.split(",")[1]);
                    JSONArray processedData = preprocessor.processForecastData(rawJson);

                    for (int i = 0; i < processedData.length(); i++) {
                        Document doc = Document.parse(processedData.getJSONObject(i).toString());
                        Bson filter = Filters.and(
                                Filters.eq("city", doc.get("city")),
                                Filters.eq("datetime", doc.get("datetime")));
                        collection.replaceOne(filter, doc, new ReplaceOptions().upsert(true));
                    }
                    LOGGER.info("Pronostico almacenado para " + cityName + " (" + processedData.length() + " entradas)");
                    succeeded++;
                } catch (Exception e) {
                    LOGGER.severe("Fallo obteniendo/guardando el pronostico de " + cityName + ": " + e.getMessage());
                    failed++;
                }
            }
        } catch (Exception e) {
            LOGGER.severe("No se pudo conectar a MongoDB, se aborta la ingesta: " + e.getMessage());
            return;
        }

        LOGGER.info("Ingesta de clima finalizada. Ciudades OK: " + succeeded + ", fallidas: " + failed);
    }
}

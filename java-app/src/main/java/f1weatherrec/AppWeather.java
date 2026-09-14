package f1weatherrec;

import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoClients;
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

/**
 * Ingesta diaria del pronostico de OpenWeatherMap (5 dias / 3 horas).
 *
 * Se pide por COORDENADAS DE CIRCUITO leidas de la coleccion `circuits`
 * (cargada desde ml/config/circuits.json), no por una lista de ciudades
 * hardcodeada: elimina el desajuste ciudad-circuito y hace que los circuitos
 * nuevos (Madrid, Sepang...) entren automaticamente al anadirlos al catalogo.
 */
public class AppWeather {
    private static final Logger LOGGER = Logger.getLogger(AppWeather.class.getName());

    public static void main(String[] args) {
        LogConfig.configure();
        LOGGER.info("Iniciando ingesta de pronosticos de clima...");

        int succeeded = 0;
        int failed = 0;

        String host = System.getenv().getOrDefault("MONGO_HOST", "localhost");
        String port = System.getenv().getOrDefault("MONGO_PORT", "27017");
        String dbName = System.getenv().getOrDefault("MONGO_DB", "F1-WeatherRec");

        try (MongoClient mongoClient = MongoClients.create("mongodb://" + host + ":" + port)) {
            MongoDatabase database = mongoClient.getDatabase(dbName);
            LOGGER.info("Conectado a MongoDB en " + host + ":" + port + " (db=" + dbName + ")");

            MongoCollection<Document> circuits = database.getCollection("circuits");
            MongoCollection<Document> collection = database.getCollection("forecast_data");
            collection.createIndex(
                    Indexes.ascending("circuit_short_name", "datetime"), new IndexOptions().unique(true));

            ForecastDataPreprocessor preprocessor = new ForecastDataPreprocessor();

            for (Document circuit : circuits.find()) {
                String name = circuit.getString("circuit_short_name");
                Double lat = circuit.getDouble("lat");
                Double lon = circuit.getDouble("lon");
                if (lat == null || lon == null) {
                    LOGGER.warning("Circuito sin coordenadas, se omite: " + name);
                    failed++;
                    continue;
                }

                try {
                    String rawJson = WeatherService.getWeatherDataByCoords(lat, lon);
                    JSONArray processedData = preprocessor.processForecastData(rawJson, name);

                    for (int i = 0; i < processedData.length(); i++) {
                        Document doc = Document.parse(processedData.getJSONObject(i).toString());
                        Bson filter = Filters.and(
                                Filters.eq("circuit_short_name", doc.get("circuit_short_name")),
                                Filters.eq("datetime", doc.get("datetime")));
                        collection.replaceOne(filter, doc, new ReplaceOptions().upsert(true));
                    }
                    LOGGER.info("Pronostico almacenado para " + name + " (" + processedData.length() + " entradas)");
                    succeeded++;
                } catch (Exception e) {
                    LOGGER.severe("Fallo obteniendo/guardando el pronostico de " + name + ": " + e.getMessage());
                    failed++;
                }
            }
        } catch (Exception e) {
            LOGGER.severe("No se pudo conectar a MongoDB, se aborta la ingesta: " + e.getMessage());
            return;
        }

        LOGGER.info("Ingesta de clima finalizada. Circuitos OK: " + succeeded + ", fallidos: " + failed);
    }
}

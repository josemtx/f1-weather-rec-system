package trabajo;

import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoClients;
import com.mongodb.client.MongoDatabase;
import com.mongodb.client.model.Filters;
import com.mongodb.client.model.IndexOptions;
import com.mongodb.client.model.Indexes;
import com.mongodb.client.model.ReplaceOptions;
import org.bson.Document;
import org.bson.conversions.Bson;
import com.mongodb.client.MongoCollection;
import org.json.JSONArray;

import java.util.logging.Logger;

public class MongoDBClient implements AutoCloseable {
    private static final Logger LOGGER = Logger.getLogger(MongoDBClient.class.getName());

    private final MongoClient client;
    private final MongoDatabase database;

    public MongoDBClient() {
        String host = System.getenv().getOrDefault("MONGO_HOST", "localhost");
        int port = Integer.parseInt(System.getenv().getOrDefault("MONGO_PORT", "27017"));
        String dbName = System.getenv().getOrDefault("MONGO_DB", "F1-WeatherRec");

        this.client = MongoClients.create("mongodb://" + host + ":" + port);
        this.database = client.getDatabase(dbName);
        try {
            ensureIndexes();
        } catch (Exception e) {
            client.close();
            throw new IllegalStateException("No se pudo conectar/preparar MongoDB en " + host + ":" + port
                    + " (db=" + dbName + "): " + e.getMessage(), e);
        }
        LOGGER.info("Conectado a MongoDB en " + host + ":" + port + " (db=" + dbName + ")");
    }

    private void ensureIndexes() {
        IndexOptions unique = new IndexOptions().unique(true);
        database.getCollection("sessions").createIndex(Indexes.ascending("session_key"), unique);
        database.getCollection("drivers").createIndex(Indexes.ascending("driver_number", "session_key"), unique);
        database.getCollection("stints").createIndex(
                Indexes.ascending("session_key", "driver_number", "stint_number"), unique);
        database.getCollection("weather").createIndex(Indexes.ascending("session_key", "date"), unique);
        database.getCollection("positions").createIndex(Indexes.ascending("driver_number", "session_key"), unique);
        database.getCollection("laps").createIndex(
                Indexes.ascending("session_key", "driver_number", "lap_number"), unique);
        database.getCollection("pit").createIndex(
                Indexes.ascending("session_key", "driver_number", "date"), unique);
    }

    public void insertSessionsData(JSONArray sessionsData) {
        MongoCollection<Document> col = database.getCollection("sessions");
        ReplaceOptions upsert = new ReplaceOptions().upsert(true);
        for (int i = 0; i < sessionsData.length(); i++) {
            Document doc = Document.parse(sessionsData.getJSONObject(i).toString());
            Bson filter = Filters.eq("session_key", doc.get("session_key"));
            col.replaceOne(filter, doc, upsert);
        }
    }

    public void insertDriversData(JSONArray driversData) {
        MongoCollection<Document> col = database.getCollection("drivers");
        ReplaceOptions upsert = new ReplaceOptions().upsert(true);
        for (int i = 0; i < driversData.length(); i++) {
            Document doc = Document.parse(driversData.getJSONObject(i).toString());
            Bson filter = Filters.and(
                    Filters.eq("driver_number", doc.get("driver_number")),
                    Filters.eq("session_key", doc.get("session_key")));
            col.replaceOne(filter, doc, upsert);
        }
    }

    public void insertStintsData(JSONArray stintsData) {
        MongoCollection<Document> col = database.getCollection("stints");
        ReplaceOptions upsert = new ReplaceOptions().upsert(true);
        for (int i = 0; i < stintsData.length(); i++) {
            Document doc = Document.parse(stintsData.getJSONObject(i).toString());
            Bson filter = Filters.and(
                    Filters.eq("session_key", doc.get("session_key")),
                    Filters.eq("driver_number", doc.get("driver_number")),
                    Filters.eq("stint_number", doc.get("stint_number")));
            col.replaceOne(filter, doc, upsert);
        }
    }

    public void insertWeatherData(JSONArray weatherData) {
        MongoCollection<Document> col = database.getCollection("weather");
        ReplaceOptions upsert = new ReplaceOptions().upsert(true);
        for (int i = 0; i < weatherData.length(); i++) {
            Document doc = Document.parse(weatherData.getJSONObject(i).toString());
            Bson filter = Filters.and(
                    Filters.eq("session_key", doc.get("session_key")),
                    Filters.eq("date", doc.get("date")));
            col.replaceOne(filter, doc, upsert);
        }
    }

    public void insertPositionsData(JSONArray positionsData) {
        MongoCollection<Document> col = database.getCollection("positions");
        ReplaceOptions upsert = new ReplaceOptions().upsert(true);
        for (int i = 0; i < positionsData.length(); i++) {
            Document doc = Document.parse(positionsData.getJSONObject(i).toString());
            Bson filter = Filters.and(
                    Filters.eq("driver_number", doc.get("driver_number")),
                    Filters.eq("session_key", doc.get("session_key")));
            col.replaceOne(filter, doc, upsert);
        }
    }

    public void insertLapsData(JSONArray lapsData) {
        MongoCollection<Document> col = database.getCollection("laps");
        ReplaceOptions upsert = new ReplaceOptions().upsert(true);
        for (int i = 0; i < lapsData.length(); i++) {
            Document doc = Document.parse(lapsData.getJSONObject(i).toString());
            Bson filter = Filters.and(
                    Filters.eq("session_key", doc.get("session_key")),
                    Filters.eq("driver_number", doc.get("driver_number")),
                    Filters.eq("lap_number", doc.get("lap_number")));
            col.replaceOne(filter, doc, upsert);
        }
    }

    public void insertPitData(JSONArray pitData) {
        MongoCollection<Document> col = database.getCollection("pit");
        ReplaceOptions upsert = new ReplaceOptions().upsert(true);
        for (int i = 0; i < pitData.length(); i++) {
            Document doc = Document.parse(pitData.getJSONObject(i).toString());
            Bson filter = Filters.and(
                    Filters.eq("session_key", doc.get("session_key")),
                    Filters.eq("driver_number", doc.get("driver_number")),
                    Filters.eq("date", doc.get("date")));
            col.replaceOne(filter, doc, upsert);
        }
    }

    @Override
    public void close() {
        client.close();
    }
}

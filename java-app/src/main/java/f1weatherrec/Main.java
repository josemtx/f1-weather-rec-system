package f1weatherrec;

public final class Main {
    private Main() {
    }

    public static void main(String[] args) {
        if (args.length != 1 || !(args[0].equals("f1") || args[0].equals("weather"))) {
            System.err.println("Uso: java -jar F1-WeatherRec.jar <f1|weather>");
            System.err.println("  f1      -> ingesta de sesiones/pilotos/stints/clima/posiciones (OpenF1)");
            System.err.println("  weather -> ingesta de pronosticos por ciudad (OpenWeatherMap)");
            System.exit(2);
        }

        if (args[0].equals("f1")) {
            AppFormula1.main(new String[0]);
        } else {
            AppWeather.main(new String[0]);
        }
    }
}

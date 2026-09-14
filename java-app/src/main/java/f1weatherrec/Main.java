package f1weatherrec;

public final class Main {
    private Main() {
    }

    public static void main(String[] args) {
        boolean f1 = args.length >= 1 && args[0].equals("f1") && (args.length == 1 || args[1].equals("incremental"));
        boolean weather = args.length == 1 && args[0].equals("weather");
        if (!f1 && !weather) {
            System.err.println("Uso: java -jar F1-WeatherRec.jar <f1 [incremental]|weather>");
            System.err.println("  f1             -> ingesta completa 2018-2026 de OpenF1 (sesiones/pilotos/stints/vueltas/boxes/clima/posiciones)");
            System.err.println("  f1 incremental -> solo el anio en curso y solo las sesiones que aun no tienen vueltas en Mongo");
            System.err.println("  weather        -> ingesta de pronosticos por circuito (OpenWeatherMap)");
            System.exit(2);
        }

        if (f1) {
            AppFormula1.main(args.length == 2 ? new String[] {"incremental"} : new String[0]);
        } else {
            AppWeather.main(new String[0]);
        }
    }
}

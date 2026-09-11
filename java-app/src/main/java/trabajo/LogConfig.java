package trabajo;

public final class LogConfig {
    private LogConfig() {
    }

    public static void configure() {
        System.setProperty("java.util.logging.SimpleFormatter.format",
                "%1$tF %1$tT [%4$-7s] %3$s - %5$s%6$n");
    }
}

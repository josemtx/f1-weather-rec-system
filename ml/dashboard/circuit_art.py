"""Silueta del circuito para la cabecera de carrera.

Los SVG vienen del paquete oficial de iconos de circuito de F1.com (carpeta
`ml/dashboard/svg/`, paquete oficial de F1.com aportado por el usuario -- ver memoria de
sesion: no se generan por telemetria, hay una fuente mas simple y mejor).
Los 25 ficheros comparten estilo exacto: viewBox 524.4x524.4, una sola clase
`.st0{fill:#241758;}` (silueta solida). Se recolorea a `currentColor` para
que herede el rojo/gris del tema por CSS, en vez de quedar fijo en el morado
original.
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

SVG_DIR = Path(__file__).resolve().parent / "svg"

# circuit_short_name (canonico, ver ml/config/circuits.json) -> fichero.
# Solo cubre los circuitos para los que el usuario aporto arte; el resto
# se queda sin dibujo (el dashboard lo trata como opcional, no como error).
CIRCUIT_SVG_FILES = {
    "Sakhir": "Bahrain.svg",
    "Jeddah": "Saudi-Arabia.svg",
    "Melbourne": "Australia.svg",
    "Suzuka": "Suzuka.svg",
    "Shanghai": "China.svg",
    "Miami": "Miami.svg",
    "Imola": "Imola.svg",
    "Monte-Carlo": "Monaco.svg",
    "Montreal": "Canada.svg",
    "Barcelona": "Barcelona.svg",
    "Spielberg": "Austria.svg",
    "Silverstone": "Britian.svg",
    "Budapest": "Hungary.svg",
    "Spa": "Belgium.svg",
    "Monza": "Monza.svg",
    "Baku": "Azerbaijan.svg",
    "Singapore": "Singapore.svg",
    "Austin": "COTA.svg",
    "Ciudad de Mexico": "Mexico.svg",
    "Sao Paulo": "Brazil.svg",
    "Las Vegas": "Las-Vegas.svg",
    "Lusail": "Qatar.svg",
    "Yas Island": "Abu-Dhabi.svg",
    "Madring": "Circuito-IFEMA-Madrid.svg",
    "Portimao": "algarve-international-circuit.svg",
}

_FILL_RE = re.compile(r"\.st0\{fill:#[0-9a-fA-F]{6};\}")


def get_circuit_svg_markup(circuit_short_name: str) -> str | None:
    """Devuelve el <svg>...</svg> recoloreado a currentColor, listo para
    inyectar por innerHTML, o None si no hay arte para ese circuito."""
    filename = CIRCUIT_SVG_FILES.get(circuit_short_name)
    if filename is None:
        return None

    path = SVG_DIR / filename
    if not path.exists():
        logger.warning(f"circuit_art: {filename} no existe en {SVG_DIR}")
        return None

    raw = path.read_text(encoding="utf-8")
    recolored = _FILL_RE.sub(".st0{fill:currentColor;}", raw)

    match = re.search(r"<svg[\s\S]*</svg>", recolored)
    if match is None:
        logger.warning(f"circuit_art: no se encontro un <svg> valido en {filename}")
        return None
    return match.group(0)

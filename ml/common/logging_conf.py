"""Formato de logging compartido para los scripts de ml/, igual que server.py."""

import logging


def configure(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

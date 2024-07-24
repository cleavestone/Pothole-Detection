import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')


list_of_files = [
    "app.py",
    "model",
    "static",
    "templates/index.html",
    "templates/result.html",
    "uploads",
    "requirements.txt"
]

for filepath in list_of_files:
    path = Path(filepath)

    if path.suffix:  # Check if path is a file
        # Create parent directories
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            logging.info(f"Creating directory: {path.parent}")

        # Create file if it doesn't exist or if it is empty
        if not path.exists() or path.stat().st_size == 0:
            path.touch()
            logging.info(f"Creating file: {path}")
    else:  # Path is a directory
        path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Creating directory: {path}")
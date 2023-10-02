import logging
from pathlib import Path
from graph_package.configs.directories import Directories

def init_logger():
    # Create a logger object
    log_path = Directories.MODULE_PATH / "logs" 
    log_path.mkdir(exist_ok=True)
    log_file = log_path / "log.txt"
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create a file handler
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.DEBUG)

    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    return logger


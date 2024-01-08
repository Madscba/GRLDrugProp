import logging
import colorlog
from pathlib import Path
from graph_package.configs.directories import Directories
import sys
from graph_package.configs.definitions import model_dict, dataset_dict 
from graph_package.configs.directories import Directories
import sys
from collections import OrderedDict



def remove_prefix_from_keys(d, prefix):
    """
    Recursively removes a prefix from the keys of an ordered dictionary and all its sub-dictionaries.

    Args:
        d (OrderedDict): The ordered dictionary to modify.
        prefix (str): The prefix to remove from the keys.

    Returns:
        OrderedDict: The modified ordered dictionary.
    """
    new_dict = OrderedDict()
    for key, value in d.items():
        new_key = key[len(prefix):] if key.startswith(prefix) else key
        new_dict[new_key] = value
    return new_dict

def init_logger():
    # Create a logger object
    logger = logging.getLogger(__name__)
    # Create a file handler
    # Create a stream handler with colored output
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    )
    # Add the handlers to the logger
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger

logger = init_logger()

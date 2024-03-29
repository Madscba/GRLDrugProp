from dataclasses import dataclass
from pathlib import Path
import graph_package


@dataclass
class Directories:
    """Class with all paths used in the repository."""

    REPO_PATH = Path(graph_package.__file__).parent.parent
    MODULE_PATH = Path(graph_package.__file__).parent
    CHECKPOINT_PATH = REPO_PATH / "checkpoints"
    CONFIG_PATH = MODULE_PATH / "configs"
    DATA_PATH = REPO_PATH / "data"
    TESTS = REPO_PATH / "tests"
    TESTS_DATA = TESTS / "data"
    LOGGING_FOLDER = REPO_PATH / "logs"
    OUTPUT_PATH = REPO_PATH / "outputs"

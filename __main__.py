"""CLI entrypoint for module."""

# Import system libraries
import argparse
import sys
from typing import List

# Import custom libraries.
from graph_package.utils.helpers import init_logger

logger = init_logger()


def parse_args(args: List) -> argparse.Namespace:
    """Argument parser."""
    parser = argparse.ArgumentParser(description="")

    # Add as many arguments as we want here.
    parser.add_argument(
        "--run",
        type=str,
        choices=["grid_search", "predict", "train"],
        help="Option of the code to run",
        default="grid_search",
    )
    parsed_args = parser.parse_args(args)

    return parsed_args


if __name__ == "__main__":
    parsed_args = parse_args(sys.argv[1:])

    if parsed_args.run == "grid_search":
        print("Trigger a grid-search")

    elif parsed_args.run == "train":
        print("Trigger a model training")

    elif parsed_args.run == "predict":
        print("Trigger a model prediction")

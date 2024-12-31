from pathlib import Path
import argparse


def config_file(filepath):
    """
    Custom argparse type for validating that a file exists and is a yaml file.

    Args:
    filepath: The file path string.

    Returns:
    A pathlib.Path object if the file is valid, otherwise raises an error.
    """
    path = Path(filepath)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File not found: {filepath}")
    if not path.suffix == ".yaml":
        raise argparse.ArgumentTypeError(f"File must be a yaml file: {filepath}")
    return path


def model_dir(filepath):
    path = Path(filepath)
    if ".." in path.parts:
        raise argparse.ArgumentTypeError("model_dir cannot contain '..'")
    if len(path.parts) > 2:
        raise argparse.ArgumentTypeError("model_dir cannot be more than 2 levels deep")
    return path

# Import necessary libraries
import os
import shutil
import sys
from pathlib import Path
from typing import List



# This script is used to remove __pycache__ folders from the given path and its subdirectories.
def remove_pycache(path: Path) -> List[Path]:
    """
    Remove __pycache__ folders from the given path and its subdirectories.
    """
    pycache_folders = list(path.rglob("__pycache__"))
    for folder in pycache_folders:
        shutil.rmtree(folder)
    return pycache_folders



if __name__ == "__main__":

    # Get the path to the app directory
    path = Path(os.path.join(os.path.dirname(os.getcwd()), "InnoTSR", "app"))
    print(f"Path to app directory: {path}")

    # Check if the path exists
    if not path.exists():
        print(f"Path {path} does not exist.")
        sys.exit(1)

    # Remove __pycache__ folders from the given path
    removed_folders = remove_pycache(path)
    if removed_folders:
        print(f"Removed __pycache__ folders: {removed_folders}")
    else:
        print("No __pycache__ folders found.")





"""Scan a directory for corrupt MP4 files and remove them.

Run this manually after an interrupted movie-generation stage to clear out
partially-written files before re-running the stage.

Usage:
    python -m src.torch.vit.cleanup_corrupt_movies --dir plots/attribution_movies
    python -m src.torch.vit.cleanup_corrupt_movies --dir plots/subset/movies/raw --dry-run

# TODO: move is_valid_mp4 to src/torch/vit/utils.py and import it here + in
#       create_movies.py, create_raw_movie.py, create_contour_grid_movie.py so
#       each script can skip already-valid outputs without manual intervention.
"""

import os
import subprocess
import argparse
from pathlib import Path


def is_valid_mp4(path: str) -> bool:
    """Return True if the file exists and ffmpeg reports no errors.

    Uses ffmpeg's null muxer to decode the entire file without writing output.
    A non-zero return code or any stderr output indicates corruption.

    Args:
        path (str): Path to the MP4 file to validate.

    Returns:
        bool: True if the file is valid, False if missing or corrupt.
    """
    if not os.path.exists(path):
        return False
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "null", "-"],
        capture_output=True,
    )
    return result.returncode == 0 and not result.stderr


def cleanup_directory(directory: str, dry_run: bool = False) -> None:
    """Scan directory for corrupt MP4s and optionally remove them.

    Args:
        directory (str): Directory to scan recursively for .mp4 files.
        dry_run (bool): If True, report corrupt files but do not delete them.
    """
    mp4_files = list(Path(directory).rglob("*.mp4"))
    if not mp4_files:
        print(f"No .mp4 files found in {directory}")
        return

    print(f"Checking {len(mp4_files)} file(s) in {directory}...")
    corrupt = []
    for path in mp4_files:
        if is_valid_mp4(str(path)):
            print(f"  OK       {path}")
        else:
            print(f"  CORRUPT  {path}")
            corrupt.append(path)

    if not corrupt:
        print("\nAll files are valid.")
        return

    print(f"\n{len(corrupt)} corrupt file(s) found.")
    if dry_run:
        print("Dry-run mode — no files deleted.")
    else:
        for path in corrupt:
            os.remove(path)
            print(f"  Deleted  {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove corrupt MP4 files from a directory before re-running a stage."
    )
    parser.add_argument("--dir", required=True, help="Directory to scan")
    parser.add_argument("--dry-run", action="store_true", help="Report only, do not delete")
    args = parser.parse_args()

    cleanup_directory(args.dir, dry_run=args.dry_run)

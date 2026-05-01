"""Generate a full-disk SDO/AIA movie for a given date range.

Downloads AIA full-disk FITS files via JSOC and renders each frame as a
solar map image, then stitches them into an MP4 via ffmpeg.

JSOC requires a registered email — register once at:
    http://jsoc.stanford.edu/ajax/register_email.html

Usage:
    python -m src.torch.vit.fulldisk_movie --start 2014-01-01 --end 2014-01-03
    python -m src.torch.vit.fulldisk_movie --start 2014-01-01 --end 2014-01-03 \
        --wavelength 304 --sample 30 --fps 10
    python -m src.torch.vit.fulldisk_movie --start 2014-01-01 --end 2014-01-03 \
        --notify you@example.com
"""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import sunpy.map
from dotenv import load_dotenv
from sunpy.net import Fido, attrs as a
from tqdm import tqdm

load_dotenv()

matplotlib.use("Agg")

DEFAULT_WAVELENGTH = 171
DEFAULT_SAMPLE_MINUTES = 60
DEFAULT_FPS = 10
DEFAULT_OUTPUT_DIR = "plots"

VALID_WAVELENGTHS = [94, 131, 171, 193, 211, 304, 335]

# AIA EUV and UV series on JSOC
_JSOC_SERIES = {
    "euv": "aia.lev1_euv_12s",
    "uv": "aia.lev1_uv_24s",
}
_UV_WAVELENGTHS = {1600, 1700}


def fetch_aia(start, end, wavelength, sample_minutes, notify_email, cache_path):
    """Download AIA full-disk FITS from JSOC for the given time range.

    On first run, fetches from JSOC and saves local paths to cache_path.
    On subsequent runs, loads paths from cache_path to skip JSOC entirely.
    Returns a list of sunpy Map objects sorted by date.
    """
    if Path(cache_path).exists():
        print(f"Loading cached file list from {cache_path} ...")
        files = [f for f in Path(cache_path).read_text().splitlines() if f]
    else:
        series = _JSOC_SERIES["uv"] if wavelength in _UV_WAVELENGTHS else _JSOC_SERIES["euv"]
        print(f"Querying JSOC ({series}) for AIA {wavelength} Å  {start} → {end}  (sample={sample_minutes} min) ...")

        result = Fido.search(
            a.Time(start, end),
            a.jsoc.Series(series),
            a.jsoc.Notify(notify_email),
            a.Wavelength(wavelength * u.angstrom),
            a.Sample(sample_minutes * u.minute),
        )
        print(result)
        if len(result["jsoc"]) == 0:
            raise RuntimeError("No AIA data found for the requested time range.")

        print(f"Fetching {len(result['jsoc'])} files from JSOC ...")
        files = Fido.fetch(result)
        if not files:
            raise RuntimeError("Fido.fetch returned no files.")

        Path(cache_path).write_text("\n".join(str(f) for f in files))
        print(f"Cached file list to {cache_path}")

    maps = [sunpy.map.Map(f) for f in files]
    maps.sort(key=lambda m: m.date)
    print(f"Loaded {len(maps)} frames.")
    return maps


def render_movie(maps, wavelength, output_path, fps):
    """Render each Map as a PNG frame and stitch into an MP4."""
    plt.ioff()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, smap in enumerate(tqdm(maps, desc="Rendering frames")):
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(projection=smap)
            smap.plot(axes=ax, clip_interval=(1, 99.9) * u.percent)
            smap.draw_limb(axes=ax)
            ax.set_title(
                f"SDO/AIA {wavelength} Å  |  {smap.date.iso}",
                fontsize=10,
                pad=6,
            )
            frame_path = os.path.join(tmpdir, f"frame_{i:04d}.png")
            fig.savefig(frame_path, bbox_inches="tight", dpi=100)
            plt.close("all")  # smap.plot() spawns stray figures; close everything

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", os.path.join(tmpdir, "frame_%04d.png"),
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                output_path,
            ],
            check=True,
        )

    print(f"\nSaved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Full-disk SDO/AIA movie for a date range.")
    parser.add_argument("--start", required=True, help="Start date e.g. 2014-01-01")
    parser.add_argument("--end", required=True, help="End date e.g. 2014-01-03")
    parser.add_argument(
        "--wavelength",
        type=int,
        default=DEFAULT_WAVELENGTH,
        choices=VALID_WAVELENGTHS,
        help=f"AIA passband in Å (default: {DEFAULT_WAVELENGTH})",
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=DEFAULT_SAMPLE_MINUTES,
        metavar="MINUTES",
        help=f"Cadence in minutes between frames (default: {DEFAULT_SAMPLE_MINUTES})",
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--notify",
        default=None,
        metavar="EMAIL",
        help="JSOC-registered email. Falls back to JSOC_EMAIL env var.",
    )
    args = parser.parse_args()

    notify_email = args.notify or os.environ.get("JSOC_EMAIL")
    if not notify_email:
        parser.error("JSOC email required: set JSOC_EMAIL in .env or pass --notify")

    start_slug = args.start.replace("-", "")
    end_slug = args.end.replace("-", "")
    slug = f"fulldisk_{args.wavelength}_{start_slug}_{end_slug}"
    output_path = str(Path(args.output_dir) / f"{slug}.mp4")
    cache_path = str(Path(args.output_dir) / f"{slug}.files")

    if Path(output_path).exists():
        print(f"Output already exists: {output_path} — skipping fetch and render.")
        return

    maps = fetch_aia(args.start, args.end, args.wavelength, args.sample, notify_email, cache_path)
    render_movie(maps, args.wavelength, output_path, args.fps)


if __name__ == "__main__":
    main()

"""Inspect raw GOES X-ray flux for a specific AARP.

Fetches GOES XRS timeseries over the AARP's observation window and either
prints a table of raw values (default) or saves an interactive Plotly HTML.

No astro_utils dependency — uses sunpy Fido directly.

Usage:
    python -m src.torch.vit.inspect_goes --aarp-id 377
    python -m src.torch.vit.inspect_goes --aarp-id 377 --start 2014-01-01 --end 2014-01-03
    python -m src.torch.vit.inspect_goes --aarp-id 377 --plot
    python -m src.torch.vit.inspect_goes --aarp-id 377 --start 2014-01-01 --end 2014-01-03 --plot
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sunpy.timeseries as sunpy_ts
from sunpy.net import Fido, attrs as a

DEFAULT_JSON_PATH = "solar_dataset.json"
DEFAULT_OUTPUT_DIR = "plots"
GOES_CHANNELS = ["xrsa", "xrsb"]
FETCH_WINDOW_HOURS = 24 * 4.5  # ±4.5 days around AARP midtime


# ==================== DATA LOADING ====================

def load_aarp_df(aarp_id, json_path):
    """Read JSON and return combined DataFrame rows for this AARP across all splits."""
    with open(json_path) as f:
        metadata = json.load(f)

    splits = {}
    for split in ["training", "validation", "test"]:
        rows = metadata.get(split, [])
        if rows:
            df = pd.DataFrame(rows)
            df["split"] = split
            splits[split] = df

    combined = pd.concat(splits.values(), ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True).dt.tz_localize(None)

    aarp_df = combined[combined["aarp_id"] == aarp_id].copy()
    if aarp_df.empty:
        raise ValueError(f"AARP {aarp_id} not found in any split of {json_path}")

    found = {s: len(df[df["aarp_id"] == aarp_id]) for s, df in splits.items()}
    found_in = [s for s, n in found.items() if n > 0]
    print(f"AARP {aarp_id}: found in {found_in} ({len(aarp_df)} frames)")

    return aarp_df


def compute_fetch_window(aarp_df):
    """Return (start, end) datetimes centred on the AARP observation midpoint."""
    timestamps = aarp_df["timestamp"]
    mid = timestamps.min() + (timestamps.max() - timestamps.min()) / 2
    delta = pd.Timedelta(hours=FETCH_WINDOW_HOURS)
    return mid - delta, mid + delta


# ==================== GOES FETCH ====================

def fetch_goes(start, end):
    """Fetch GOES XRS 1-second data between start and end (datetime or str)."""
    start_str = pd.Timestamp(start).isoformat()
    end_str = pd.Timestamp(end).isoformat()
    print(f"Fetching GOES data {start_str} → {end_str} ...")

    result = Fido.search(
        a.Time(start_str, end_str),
        a.Instrument("XRS"),
        a.Resolution("flx1s"),
    )
    responses = result["xrs"]
    satellite_idx = int(np.argmax(responses["SatelliteNumber"]))
    files = Fido.fetch(result[0, satellite_idx:])
    goes_ts = sunpy_ts.TimeSeries(files, source="XRS", concatenate=True)
    return goes_ts.truncate(start_str, end_str)


# ==================== FILTER ====================

def filter_goes(goes_ts, start=None, end=None):
    """Return a DataFrame from XRSTimeSeries, optionally truncated to [start, end]."""
    if start or end:
        df = goes_ts.data[GOES_CHANNELS].copy()
        a_ = pd.Timestamp(start) if start else df.index[0]
        b_ = pd.Timestamp(end) if end else df.index[-1]
        return goes_ts.truncate(a_, b_).data[GOES_CHANNELS].copy()
    return goes_ts.data[GOES_CHANNELS].copy()


# ==================== TABLE ====================

def print_table(df, aarp_id, start=None, end=None):
    label = f"AARP {aarp_id}"
    if start or end:
        label += f"  [{start or '...'} → {end or '...'}]"

    print(f"\nRaw GOES flux — {label}")
    print(f"  Rows : {len(df)}")
    for ch in GOES_CHANNELS:
        print(f"  {ch}  min={df[ch].min():.3e}  max={df[ch].max():.3e}")

    print()
    for ch in GOES_CHANNELS:
        flat = df[ch].diff().eq(0).sum()
        if flat > 0:
            print(f"  [{ch}] {flat} consecutive repeated values — possible clipping/saturation")

    print()
    pd.set_option("display.float_format", "{:.4e}".format)
    pd.set_option("display.max_rows", None)
    print(df.to_string())


# ==================== PLOTLY ====================

def save_plotly_html(df, aarp_id, output_dir, start=None, end=None):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed — run: pip install plotly")
        return

    colors = {"xrsa": "blue", "xrsb": "red"}
    labels = {"xrsa": "0.5–4.0 Å (xrsa)", "xrsb": "1.0–8.0 Å (xrsb)"}

    fig = go.Figure()
    for ch in GOES_CHANNELS:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[ch],
            mode="lines",
            name=labels[ch],
            line=dict(color=colors[ch], width=1),
            hovertemplate="%{x}<br>%{y:.3e} W/m²<extra>" + labels[ch] + "</extra>",
        ))

    fig.update_layout(
        title=f"GOES X-ray flux — AARP {aarp_id}",
        xaxis_title="Time (UTC)",
        yaxis_title="Flux (W m⁻²)",
        yaxis_type="log",
        yaxis=dict(exponentformat="e", showexponent="all"),
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99),
        template="plotly_white",
    )

    suffix = f"_{start}_{end}" if (start or end) else ""
    out_path = Path(output_dir) / f"goes_inspect_{aarp_id}{suffix}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))
    print(f"\n✓ Saved interactive plot to {out_path}")


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description="Inspect raw GOES flux for an AARP.")
    parser.add_argument("--aarp-id", type=int, required=True)
    parser.add_argument("--json-path", default=DEFAULT_JSON_PATH)
    parser.add_argument("--start", default=None, help="Filter start e.g. 2014-01-01")
    parser.add_argument("--end", default=None, help="Filter end e.g. 2014-01-03")
    parser.add_argument("--plot", action="store_true", help="Save interactive Plotly HTML")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    aarp_df = load_aarp_df(args.aarp_id, args.json_path)
    fetch_start, fetch_end = compute_fetch_window(aarp_df)
    goes_ts = fetch_goes(fetch_start, fetch_end)
    df = filter_goes(goes_ts, start=args.start, end=args.end)

    print_table(df, args.aarp_id, start=args.start, end=args.end)

    if args.plot:
        save_plotly_html(df, args.aarp_id, args.output_dir, start=args.start, end=args.end)


if __name__ == "__main__":
    main()

"""Inspect raw GOES X-ray flux for a given date range.

Fetches GOES XRS 1-second timeseries and either prints a table of raw values
(default) or saves an interactive Plotly HTML for pan/zoom inspection.

Usage:
    python -m src.torch.vit.inspect_goes --start 2014-01-01 --end 2014-01-03
    python -m src.torch.vit.inspect_goes --start 2014-01-01 --end 2014-01-03 --plot
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import sunpy.timeseries as sunpy_ts
from sunpy.net import Fido, attrs as a

DEFAULT_OUTPUT_DIR = "plots"
GOES_CHANNELS = ["xrsa", "xrsb"]


def fetch_goes(start, end):
    """Fetch GOES XRS 1-second data between start and end."""
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


def print_table(df, start, end):
    print(f"\nRaw GOES flux  [{start} → {end}]")
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


def save_plotly_html(df, start, end, output_dir):
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
        title=f"GOES X-ray flux  {start} → {end}",
        xaxis_title="Time (UTC)",
        yaxis_title="Flux (W m⁻²)",
        yaxis_type="log",
        yaxis=dict(exponentformat="e", showexponent="all"),
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99),
        template="plotly_white",
    )

    start_slug = start.replace("-", "")
    end_slug = end.replace("-", "")
    out_path = Path(output_dir) / f"goes_inspect_{start_slug}_{end_slug}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))
    print(f"\n✓ Saved interactive plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Inspect raw GOES X-ray flux for a date range.")
    parser.add_argument("--start", required=True, help="Start date e.g. 2014-01-01")
    parser.add_argument("--end", required=True, help="End date e.g. 2014-01-03")
    parser.add_argument("--plot", action="store_true", help="Save interactive Plotly HTML")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    goes_ts = fetch_goes(args.start, args.end)
    df = goes_ts.data[GOES_CHANNELS].copy()

    print_table(df, args.start, args.end)

    if args.plot:
        save_plotly_html(df, args.start, args.end, args.output_dir)


if __name__ == "__main__":
    main()

"""Render per-experiment HTML reports from the papermill template.

Executes notebooks/experiment_report.ipynb once per experiment (parameters:
EXP_ID), then converts each executed notebook to reports/<exp-id>.html.
The template only READS artifacts/<exp-id>/ and metrics/<exp-id>/ — run
`dvc repro` first so those are current; missing artifacts render as
"not computed yet" warnings rather than failing the report.

Usage:
    python -m src.paper.render_reports                 # all tier_core experiments
    python -m src.paper.render_reports cv-fold0-7ch-normfix-l1-log [more-ids...]
"""

import argparse
import subprocess
import sys
from pathlib import Path

import papermill as pm
import yaml

TEMPLATE = Path("notebooks/experiment_report.ipynb")
REGISTRY_PATH = Path("paper/experiments.yaml")
REPORTS_DIR = Path("reports")
EXECUTED_DIR = REPORTS_DIR / "executed"


def render(exp_id: str) -> Path:
    EXECUTED_DIR.mkdir(parents=True, exist_ok=True)
    executed_nb = EXECUTED_DIR / f"{exp_id}.ipynb"
    print(f"[render_reports] executing template for {exp_id} ...")
    pm.execute_notebook(
        str(TEMPLATE),
        str(executed_nb),
        parameters={"EXP_ID": exp_id},
        cwd=".",  # template reads repo-relative paths (paper/, artifacts/, metrics/)
    )
    subprocess.run(
        [sys.executable, "-m", "nbconvert", "--to", "html",
         "--output-dir", str(REPORTS_DIR), str(executed_nb)],
        check=True,
    )
    html = REPORTS_DIR / f"{exp_id}.html"
    print(f"[render_reports] wrote {html}")
    return html


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("exp_ids", nargs="*",
                        help="Experiment IDs from paper/experiments.yaml specs. "
                             "Default: every tier_core experiment.")
    args = parser.parse_args()

    registry = yaml.safe_load(REGISTRY_PATH.read_text())
    known = set(registry["specs"])
    exp_ids = args.exp_ids or list(registry["tier_core"])
    unknown = [e for e in exp_ids if e not in known]
    if unknown:
        parser.error(f"Unknown experiment id(s) {unknown}. Known: {sorted(known)}")

    for exp_id in exp_ids:
        render(exp_id)


if __name__ == "__main__":
    main()

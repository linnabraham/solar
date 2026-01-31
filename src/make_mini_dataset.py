#!/usr/bin/env python3

import json
import shutil
from pathlib import Path
from collections import defaultdict

# ============================================================
# Paths (repo-anchored)
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[1]

SRC_DATASET_JSON = REPO_ROOT / "solar_dataset.json"
SRC_STATS_JSON   = REPO_ROOT / "stats.json"

SRC_ROOT = REPO_ROOT / "data" / "E8"

OUT_ROOT = REPO_ROOT / "tests" / "data" / "mini_dataset"

N_CHANNELS = 7  # "0" .. "6"

# ============================================================
# Sampling policy per split
# ============================================================
SPLIT_POLICY = {
    "training":   dict(max_aarps=2, samples_per_aarp=2),
    "validation": dict(max_aarps=1, samples_per_aarp=1),
    "test":       dict(max_aarps=1, samples_per_aarp=1),
}

# ============================================================
# Helpers
# ============================================================
def reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def group_by_label_and_aarp(samples):
    groups = defaultdict(list)
    for s in samples:
        groups[(s["label"], s["aarp_id"])].append(s)
    return groups


def select_samples(samples, max_aarps, samples_per_aarp, banned_aarps):
    """
    Select samples ensuring no AARP leakage.
    """
    groups = group_by_label_and_aarp(samples)
    selected = []
    used_aarps = set()

    for label in (1, 0):  # pos, neg
        aarps = sorted(
            aarp for (lbl, aarp) in groups
            if lbl == label and aarp not in banned_aarps
        )[:max_aarps]

        for aarp in aarps:
            selected.extend(groups[(label, aarp)][:samples_per_aarp])
            used_aarps.add(aarp)

    return selected, used_aarps


def copy_and_rewrite_sample(sample):
    new_sample = sample.copy()

    for ch in map(str, range(N_CHANNELS)):
        src = REPO_ROOT / sample[ch]
        if not src.exists():
            raise FileNotFoundError(src)

        rel = src.relative_to(SRC_ROOT)
        dst = OUT_ROOT / rel

        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)

        new_sample[ch] = str(dst.relative_to(REPO_ROOT))

    return new_sample


# ============================================================
# Main
# ============================================================
def main():
    print("Creating multi-split mini dataset…")

    reset_dir(OUT_ROOT)

    with open(SRC_DATASET_JSON) as f:
        dataset = json.load(f)

    used_aarps_global = set()
    mini_dataset = {
        "name": dataset["name"] + " (mini)",
        "description": (
            "Mini dataset for unit tests and inspection. "
            "NOT for training or evaluation."
        ),
        "channels": dataset["channels"],
    }

    # --------------------------------------------------------
    # Process each split independently
    # --------------------------------------------------------
    for split, policy in SPLIT_POLICY.items():
        print(f"\nProcessing split: {split}")

        samples = dataset.get(split)
        if samples is None:
            print(f"  [skip] split not found")
            continue

        selected, used = select_samples(
            samples,
            max_aarps=policy["max_aarps"],
            samples_per_aarp=policy["samples_per_aarp"],
            banned_aarps=used_aarps_global,
        )

        used_aarps_global |= used

        print(f"  selected samples: {len(selected)}")
        print(f"  AARPs: {sorted(used)}")

        mini_dataset[split] = [
            copy_and_rewrite_sample(s) for s in selected
        ]

    # --------------------------------------------------------
    # Write mini solar_dataset.json
    # --------------------------------------------------------
    with open(OUT_ROOT / "solar_dataset.json", "w") as f:
        json.dump(mini_dataset, f, indent=2)

    # --------------------------------------------------------
    # Copy stats.json verbatim
    # --------------------------------------------------------
    shutil.copy2(SRC_STATS_JSON, OUT_ROOT / "stats.json")

    # --------------------------------------------------------
    # README
    # --------------------------------------------------------
    (OUT_ROOT / "README.md").write_text(
        "# Mini Solar Flare Dataset\n\n"
        "- Train / validation / test splits\n"
        "- No AARP leakage across splits\n"
        "- Same schema as full dataset\n"
        "- Uses original normalization stats\n"
        "- For unit tests & collaboration only\n"
    )

    print("\nMini dataset written to:")
    print(f"  {OUT_ROOT}")


if __name__ == "__main__":
    main()

"""Build stratified, AARP-level k-fold cross-validation splits.

Pools all AARPs currently spread across the training/validation/test keys of the source
dataset JSON, groups by AARP (not by sample — an AARP's images always share one label),
and produces `n_splits` per-fold dataset JSONs. In each fold's JSON:
  - `test`       = the held-out fold's AARPs (never trained or checkpoint-selected on)
  - `training`   = remaining AARPs, minus the ones carved into `validation`
  - `validation` = a small AARP-level slice of the remaining AARPs, used only for
                    checkpoint selection (SaveBestModel) during that fold's training run

Usage:
    python -m src.torch.vit.make_cv_folds --json-path solar_dataset.json \\
        --output-dir cv_folds --n-splits 5
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def build_folds(json_path, output_dir, n_splits=5, val_frac=0.15, seed=42):
    with open(json_path, 'r') as f:
        data = json.load(f)

    all_items = data['training'] + data['validation'] + data['test']
    by_aarp = defaultdict(list)
    label_by_aarp = {}
    for item in all_items:
        by_aarp[item['aarp_id']].append(item)
        label_by_aarp[item['aarp_id']] = item['label']

    aarp_ids = np.array(sorted(by_aarp.keys()))
    labels = np.array([label_by_aarp[a] for a in aarp_ids])

    print(f"Pooled {len(aarp_ids)} AARPs ({(labels == 1).sum()} positive, "
          f"{(labels == 0).sum()} negative), {len(all_items)} total samples")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    os.makedirs(output_dir, exist_ok=True)

    fold_summary = []
    for fold_idx, (train_val_pos, test_pos) in enumerate(skf.split(aarp_ids, labels)):
        train_val_aarps = aarp_ids[train_val_pos]
        train_val_labels = labels[train_val_pos]
        test_aarps = aarp_ids[test_pos]

        # Carve a small AARP-level validation slice out of the remaining AARPs, stratified
        train_aarps, val_aarps = train_test_split(
            train_val_aarps, test_size=val_frac, stratify=train_val_labels, random_state=seed
        )

        fold_data = {
            "name": data.get("name", "solar_dataset") + f"_cv_fold{fold_idx}",
            "description": f"Stratified AARP-level CV fold {fold_idx}/{n_splits}. "
                            f"test = held-out fold, training/validation = remaining AARPs.",
            "channels": data.get("channels"),
            "training": [it for a in train_aarps for it in by_aarp[a]],
            "validation": [it for a in val_aarps for it in by_aarp[a]],
            "test": [it for a in test_aarps for it in by_aarp[a]],
        }
        fold_data["size"] = {k: len(fold_data[k]) for k in ("training", "validation", "test")}

        out_path = os.path.join(output_dir, f"fold_{fold_idx}.json")
        with open(out_path, 'w') as f:
            json.dump(fold_data, f)

        def aarp_counts(aarps):
            n_pos = sum(1 for a in aarps if label_by_aarp[a] == 1)
            return f"{len(aarps)} AARPs ({n_pos} pos, {len(aarps) - n_pos} neg)"

        summary = {
            "fold": fold_idx,
            "train_aarps": aarp_counts(train_aarps),
            "val_aarps": aarp_counts(val_aarps),
            "test_aarps": aarp_counts(test_aarps),
            "test_aarp_ids": sorted(int(a) for a in test_aarps),
            "train_samples": len(fold_data["training"]),
            "val_samples": len(fold_data["validation"]),
            "test_samples": len(fold_data["test"]),
        }
        fold_summary.append(summary)
        print(f"fold {fold_idx}: train={summary['train_aarps']}, val={summary['val_aarps']}, "
              f"test={summary['test_aarps']} -> {out_path}")
        print(f"          test AARP IDs: {summary['test_aarp_ids']}")

    with open(os.path.join(output_dir, "fold_summary.json"), 'w') as f:
        json.dump(fold_summary, f, indent=2)

    return fold_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", default="solar_dataset.json")
    parser.add_argument("--output-dir", default="cv_folds")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--val-frac", type=float, default=0.15,
                         help="Fraction of non-test AARPs (by AARP count) reserved for checkpoint-selection validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_folds(args.json_path, args.output_dir, args.n_splits, args.val_frac, args.seed)

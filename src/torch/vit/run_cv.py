"""Sequentially train + evaluate one model per AARP-level CV fold.

Resumable: skips training for any fold whose checkpoint already exists, and skips
evaluation for any fold whose result is already in the results file. Meant to run as one
long background job across all folds, one at a time (single shared GPU).

Usage:
    python -m src.torch.vit.run_cv --folds-dir cv_folds --n-splits 5 \\
        --channels 94 131 --epochs 20 --run-prefix cv-fold
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time


def run_and_log(cmd, log_path):
    print(f"$ {' '.join(cmd)}")
    with open(log_path, 'w') as log_f:
        proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    return proc.returncode


def parse_test_metrics(log_path):
    text = open(log_path).read()
    def grab(pattern):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None
    return {
        "precision": grab(r"Precision:\s*([\d.]+)"),
        "recall": grab(r"Recall:\s*([\d.]+)"),
        "accuracy": grab(r"Accuracy:\s*([\d.]+)"),
        "TP": grab(r"True Positives \(TP\):\s*(\d+)"),
        "FP": grab(r"False Positives \(FP\):\s*(\d+)"),
        "FN": grab(r"False Negatives \(FN\):\s*(\d+)"),
        "TN": grab(r"True Negatives \(TN\):\s*(\d+)"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds-dir", default="cv_folds")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--stats-file", default="stats.pkl")
    parser.add_argument("--channels", type=int, nargs="+", default=[94, 131])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--memory-threshold", type=int, default=5000)
    parser.add_argument("--run-prefix", default="cv-fold")
    parser.add_argument("--results-file", default=None)
    parser.add_argument("--log-dir", default="logs/cv")
    args = parser.parse_args()

    results_path = args.results_file or os.path.join(args.folds_dir, "cv_results.json")
    os.makedirs(args.log_dir, exist_ok=True)

    results = {}
    if os.path.exists(results_path):
        results = json.load(open(results_path))

    for i in range(args.n_splits):
        run_name = f"{args.run_prefix}{i}-{''.join(str(c) for c in args.channels)}ch"
        fold_json = os.path.join(args.folds_dir, f"fold_{i}.json")
        ckpt = os.path.join("outputs", run_name, "trained_model.pth")

        print(f"\n{'='*60}\nFold {i}: run_name={run_name}\n{'='*60}")

        if not os.path.exists(ckpt):
            train_log = os.path.join(args.log_dir, f"train_fold{i}.log")
            t0 = time.time()
            rc = run_and_log([
                sys.executable, "-m", "src.torch.vit.train",
                "--json-path", fold_json,
                "--stats-file", args.stats_file,
                "--channels", *[str(c) for c in args.channels],
                "--epochs", str(args.epochs),
                "--memory-threshold", str(args.memory_threshold),
                "--run-name", run_name,
            ], train_log)
            elapsed_min = (time.time() - t0) / 60
            print(f"  train exit={rc}, {elapsed_min:.1f} min, log={train_log}")
            if rc != 0 or not os.path.exists(ckpt):
                print(f"  FAILED — checkpoint not found at {ckpt}, skipping eval for this fold")
                results[str(i)] = {"status": "train_failed"}
                json.dump(results, open(results_path, 'w'), indent=2)
                continue
        else:
            print(f"  checkpoint already exists at {ckpt}, skipping training")

        if str(i) in results and results[str(i)].get("status") == "done":
            print("  eval already recorded, skipping")
            continue

        test_log = os.path.join(args.log_dir, f"test_fold{i}.log")
        rc = run_and_log([
            sys.executable, "-m", "src.torch.vit.test",
            "--subset", "test",
            "--trained-model", ckpt,
            "--stats-file", args.stats_file,
            "--channels", *[str(c) for c in args.channels],
            "--json-path", fold_json,
        ], test_log)
        metrics = parse_test_metrics(test_log)
        metrics["status"] = "done" if rc == 0 else "eval_failed"
        metrics["run_name"] = run_name
        results[str(i)] = metrics
        json.dump(results, open(results_path, 'w'), indent=2)
        print(f"  fold {i} result: {metrics}")

    print(f"\nAll folds processed. Results at {results_path}")
    accs = [r["accuracy"] for r in results.values() if r.get("accuracy") is not None]
    if accs:
        import statistics
        print(f"Mean test accuracy across {len(accs)} folds: "
              f"{statistics.mean(accs):.3f} +/- {statistics.stdev(accs):.3f}" if len(accs) > 1
              else f"{accs[0]:.3f}")


if __name__ == "__main__":
    main()

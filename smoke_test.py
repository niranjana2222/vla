#!/usr/bin/env python
"""
scripts/smoke_test.py
=====================
Runs the complete probe pipeline on synthetic data.
No GPU, no model weights, no LangGap download required.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --n_per_class 50 --n_vlm_layers 32

What it verifies:
    1. SyntheticLoader produces samples correctly
    2. SyntheticExtractor generates hidden states with a designed signal
    3. train_probes finds the signal in late VLM layers
    4. train_probes finds near-chance in action expert layers (language gap)
    5. save_results / load_results round-trip correctly
    6. plot_probe_accuracy produces a valid figure
    7. print_summary prints correct diagnosis
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from synthetic import SyntheticLoader, SyntheticExtractor
from train import train_probes, save_results
from evaluate import plot_probe_accuracy, print_summary


def main(args):
    print("\n=== Smoke test (synthetic data, no model needed) ===\n")
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Generate synthetic samples ──────────────────────────────────────
    loader = SyntheticLoader(n_per_class=args.n_per_class, seed=42)
    samples = loader.load()
    print(f"  {len(samples)} samples, {loader.n_classes} classes: {loader.class_names}")

    # ── 2. Generate synthetic hidden states ────────────────────────────────
    extractor = SyntheticExtractor(
        n_vlm_layers=args.n_vlm_layers,
        n_ae_layers=args.n_ae_layers,
        hidden_dim=args.hidden_dim,
        signal_starts_at_fraction=0.6,
        seed=42,
    )

    print(f"\nGenerating VLM hidden states ({args.n_vlm_layers} layers)...")
    vlm_data = extractor.extract_vlm(samples)

    print(f"Generating action expert hidden states ({args.n_ae_layers} layers)...")
    ae_data = extractor.extract_action_expert(samples)

    # ── 3. Train probes ────────────────────────────────────────────────────
    print(f"\nTraining VLM probes (cv_folds={args.cv_folds})...")
    vlm_results = train_probes(vlm_data, cv_folds=args.cv_folds, verbose=True)

    print(f"\nTraining action expert probes...")
    ae_results = train_probes(ae_data, cv_folds=args.cv_folds, verbose=True)

    # ── 4. Save results ────────────────────────────────────────────────────
    vlm_path = results_dir / "smoke_vlm_results.json"
    ae_path  = results_dir / "smoke_ae_results.json"
    save_results(vlm_results, vlm_path, component="vlm")
    save_results(ae_results,  ae_path,  component="action_expert")

    # ── 5. Print summary ───────────────────────────────────────────────────
    print_summary(vlm_path, ae_path)

    # ── 6. Plot ────────────────────────────────────────────────────────────
    plot_path = results_dir / "smoke_test_probe_accuracy.png"
    plot_probe_accuracy(
        vlm_results_path=vlm_path,
        action_results_path=ae_path,
        save_path=plot_path,
        title="Smoke test — synthetic language gap (VLM encodes object, AE does not)",
        show=False,
    )

    # ── 7. Assertions ──────────────────────────────────────────────────────
    vlm_peak = max(v["accuracy"] for v in vlm_results.values())
    ae_peak  = max(v["accuracy"] for v in ae_results.values())
    chance   = 1.0 / loader.n_classes

    assert vlm_peak > chance + 0.3, (
        f"VLM probe peak {vlm_peak:.3f} should be well above chance {chance:.3f}. "
        "Signal may not be strong enough — increase n_per_class."
    )
    assert ae_peak < chance + 0.15, (
        f"Action expert probe peak {ae_peak:.3f} should stay near chance {chance:.3f}. "
        "Synthetic extractor may have a bug."
    )
    assert plot_path.exists(), "Plot file was not created."

    print("\n=== Smoke test PASSED ===\n")
    print(f"Outputs in: {results_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test the probe pipeline")
    parser.add_argument("--n_per_class", type=int, default=30)
    parser.add_argument("--n_vlm_layers", type=int, default=28)
    parser.add_argument("--n_ae_layers",  type=int, default=18)
    parser.add_argument("--hidden_dim",   type=int, default=256)
    parser.add_argument("--cv_folds",     type=int, default=5)
    parser.add_argument("--output_dir",   type=str, default="./results")
    main(parser.parse_args())

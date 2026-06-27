#!/usr/bin/env python
"""
scripts/run_vlm_probe.py
========================
Extracts PaliGemma hidden states on LangGap scenes and trains probes.

Usage:
    python scripts/run_vlm_probe.py \
        --model google/paligemma-3b-pt-224 \
        --langgap_dir ./langgap/data \
        --output_dir ./results/vlm \
        --device cuda

    # Re-use cached hidden states (skip re-extraction):
    python scripts/run_vlm_probe.py \
        --output_dir ./results/vlm \
        --skip_extraction

    # Only a subset of scenes (faster for testing):
    python scripts/run_vlm_probe.py \
        --model google/paligemma-3b-pt-224 \
        --langgap_dir ./langgap/data \
        --scene_ids 0 1 2 3 4 \
        --output_dir ./results/vlm_small
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from langgap_loader import LangGapLoader
from vlm_extractor import VLMExtractor
from train import train_probes, save_results
from evaluate import plot_probe_accuracy, print_summary


def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    states_path = out / "hidden_states.npz"
    labels_path = out / "labels.npy"

    # ── 1. Extract (or load cached) hidden states ──────────────────────────
    if args.skip_extraction and states_path.exists():
        print("Loading cached hidden states...")
        data = np.load(states_path)
        layer_data = {int(k): (data[k], np.load(labels_path)) for k in data.files}
    else:
        scene_ids = args.scene_ids if args.scene_ids else None
        loader = LangGapLoader(
            langgap_dir=args.langgap_dir,
            scene_ids=scene_ids,
            max_per_scene=args.n_objects,
            image_size=224,
        )
        samples = loader.load()

        extractor = VLMExtractor(
            model_name=args.model,
            device=args.device,
        )
        layer_data = extractor.extract(samples)

        if args.save_states:
            print("Saving hidden states...")
            np.savez(states_path, **{str(k): X for k, (X, _) in layer_data.items()})
            np.save(labels_path, list(layer_data.values())[0][1])
            print(f"  Saved → {states_path}")

    # ── 2. Train probes ────────────────────────────────────────────────────
    print(f"\nTraining probes (cv_folds={args.cv_folds})...")
    results = train_probes(
        layer_data,
        cv_folds=args.cv_folds,
        max_iter=args.max_iter,
        verbose=True,
    )

    results_path = out / "vlm_probe_results.json"
    save_results(results, results_path, component="vlm")

    # ── 3. Evaluate ────────────────────────────────────────────────────────
    print_summary(vlm_results_path=results_path)
    plot_probe_accuracy(
        vlm_results_path=results_path,
        save_path=out / "vlm_probe_accuracy.png",
        title=f"VLM probe accuracy — {args.model}",
        show=not args.no_show,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="google/paligemma-3b-pt-224")
    p.add_argument("--langgap_dir", default="./langgap/data")
    p.add_argument("--output_dir",  default="./results/vlm")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--n_objects",   type=int, default=3)
    p.add_argument("--scene_ids",   type=int, nargs="*", default=None)
    p.add_argument("--cv_folds",    type=int, default=5)
    p.add_argument("--max_iter",    type=int, default=1000)
    p.add_argument("--save_states", action="store_true", default=True)
    p.add_argument("--skip_extraction", action="store_true")
    p.add_argument("--no_show",     action="store_true")
    main(p.parse_args())

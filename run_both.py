#!/usr/bin/env python
"""
scripts/run_both.py
===================
Overlays VLM and action expert probe accuracy curves on one plot.
Can either run full extraction or use pre-computed result JSONs.

Usage — from pre-computed results:
    python scripts/run_both.py \
        --vlm_results   ./results/vlm/vlm_probe_results.json \
        --action_results ./results/action/ae_probe_results.json \
        --output_dir    ./results

Usage — full end-to-end:
    python scripts/run_both.py \
        --model google/paligemma-3b-pt-224 \
        --action_ckpt /path/to/action_expert.pt \
        --langgap_dir ./langgap/data \
        --output_dir  ./results \
        --device cuda
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evaluate import plot_probe_accuracy, print_summary


def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    vlm_path = args.vlm_results
    ae_path  = args.action_results

    # If no pre-computed results, run extraction first
    if vlm_path is None:
        from langgap_loader import LangGapLoader
        from vlm_extractor import VLMExtractor
        from action_extractor import ActionExpertExtractor
        from train import train_probes, save_results

        loader = LangGapLoader(langgap_dir=args.langgap_dir, max_per_scene=3)
        samples = loader.load()

        vlm_ext = VLMExtractor(model_name=args.model, device=args.device)
        vlm_data = vlm_ext.extract(samples)
        vlm_results = train_probes(vlm_data, cv_folds=args.cv_folds)
        vlm_path = out / "vlm_probe_results.json"
        save_results(vlm_results, vlm_path, component="vlm")

        if args.action_ckpt:
            ae_ext = ActionExpertExtractor(
                action_expert_ckpt=args.action_ckpt,
                arch=args.arch,
                vlm_extractor=vlm_ext,
                device=args.device,
            )
            ae_data = ae_ext.extract(samples)
            ae_results = train_probes(ae_data, cv_folds=args.cv_folds)
            ae_path = out / "ae_probe_results.json"
            save_results(ae_results, ae_path, component="action_expert")

    print_summary(vlm_path, ae_path)

    plot_probe_accuracy(
        vlm_results_path=vlm_path,
        action_results_path=ae_path,
        save_path=out / "probe_accuracy_overlay.png",
        title="Object identity probe accuracy — VLM vs action expert",
        show=not args.no_show,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vlm_results",    default=None)
    p.add_argument("--action_results", default=None)
    p.add_argument("--model",          default="google/paligemma-3b-pt-224")
    p.add_argument("--action_ckpt",    default=None)
    p.add_argument("--arch",           default="smolvla")
    p.add_argument("--langgap_dir",    default="./langgap/data")
    p.add_argument("--output_dir",     default="./results")
    p.add_argument("--device",         default="cuda")
    p.add_argument("--cv_folds",       type=int, default=5)
    p.add_argument("--no_show",        action="store_true")
    main(p.parse_args())

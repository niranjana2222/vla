#!/usr/bin/env python
"""
Cross-pairing experiment: does PaliGemma encode the visual scene or the
instruction when they conflict?

Setup:
    Take an image from task i, pair it with the instruction from task j (j != i).
    The hidden states now receive conflicting signals: the image says "task i",
    the instruction says "task j".

    Train two linear probes on the same extracted hidden states:
        - Visual probe:   label = image's original task  (i)
        - Language probe: label = instruction's task     (j)

    If language probe >> visual probe → VLM follows instruction over image
    If visual probe >> language probe → VLM follows image over instruction
    If both are high → VLM encodes both independently
    If both are low  → VLM fuses them in a way neither probe can read separately

Usage:
    python run_cross_probe.py --langgap_dir ./langgap_hf --device cuda --no_show
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

sys.path.insert(0, str(Path(__file__).parent))

from langgap_loader import LangGapLoader
from vlm_extractor import VLMExtractor
from train import train_probes, save_results


def make_cross_pairs(samples: list[dict], seed: int = 42):
    """
    Pair each image from task i with the instruction from task (i+1) % n_tasks.

    Returns:
        cross_samples:       list of {image, instruction, label=image_task}
        image_labels:        int array, shape (n,)  — visual task
        instruction_labels:  int array, shape (n,)  — instruction task
    """
    by_task = defaultdict(list)
    for s in samples:
        by_task[s["label"]].append(s)

    tasks = sorted(by_task.keys())
    n = len(tasks)
    rng = random.Random(seed)

    cross_samples = []
    image_labels = []
    instruction_labels = []

    for i, task_i in enumerate(tasks):
        task_j = tasks[(i + 1) % n]
        imgs = by_task[task_i]
        instrs = by_task[task_j][:]
        rng.shuffle(instrs)
        instrs = instrs[: len(imgs)]

        for img_s, instr_s in zip(imgs, instrs):
            cross_samples.append({
                "image": img_s["image"],
                "instruction": instr_s["instruction"],
                "label": task_i,   # unused by extractor logic; we replace y below
            })
            image_labels.append(task_i)
            instruction_labels.append(task_j)

    return cross_samples, np.array(image_labels), np.array(instruction_labels)


def plot_cross_probe(
    visual_results: dict,
    language_results: dict,
    save_path: Path,
    show: bool = False,
):
    chance = list(visual_results.values())[0]["chance"]
    layers_v = sorted(visual_results)
    layers_l = sorted(language_results)

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(
        layers_v,
        [visual_results[l]["accuracy"] for l in layers_v],
        "o-", color="#534AB7", linewidth=2.2, markersize=4.5,
        label="Visual probe (predicts image's task)",
    )
    ax.plot(
        layers_l,
        [language_results[l]["accuracy"] for l in layers_l],
        "s--", color="#D85A30", linewidth=2.2, markersize=4.5,
        label="Language probe (predicts instruction's task)",
    )
    ax.axhline(
        chance, color="#888780", linestyle=":", linewidth=1.2,
        label=f"Chance ({1/chance:.0f} classes = {chance:.2f})",
    )

    ax.set_xlabel("Layer index", fontsize=12)
    ax.set_ylabel("CV probe accuracy", fontsize=12)
    ax.set_title(
        "Cross-pairing: visual vs language encoding in PaliGemma\n"
        "(image from task i, instruction from task j)",
        fontsize=12, pad=10,
    )
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, fontsize=11)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {save_path}")

    if show:
        plt.show()
    plt.close(fig)


def print_cross_summary(visual_results: dict, language_results: dict):
    vis_peak = max(v["accuracy"] for v in visual_results.values())
    lang_peak = max(v["accuracy"] for v in language_results.values())
    chance = list(visual_results.values())[0]["chance"]

    print("\n" + "=" * 60)
    print("CROSS-PAIRING RESULTS")
    print("=" * 60)
    print(f"  Visual probe peak:   {vis_peak:.3f}  (chance={chance:.3f})")
    print(f"  Language probe peak: {lang_peak:.3f}  (chance={chance:.3f})")
    print()

    gap = lang_peak - vis_peak
    if lang_peak > vis_peak + 0.05:
        print(f"  → Language probe wins by {gap:+.3f}")
        print("    PaliGemma hidden states encode the INSTRUCTION more than the image.")
        print("    VLM appears to follow language over visual context.")
    elif vis_peak > lang_peak + 0.05:
        print(f"  → Visual probe wins by {-gap:+.3f}")
        print("    PaliGemma hidden states encode the IMAGE more than the instruction.")
        print("    VLM appears to follow visual context over language.")
    else:
        print(f"  → Probes roughly tied (gap={gap:+.3f})")
        print("    VLM encodes both image and instruction similarly,")
        print("    or the conflict is resolved in a way neither probe separates cleanly.")
    print("=" * 60 + "\n")


def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Load samples ────────────────────────────────────────────────────
    print("Loading samples (tasks 0-3: black bowl pickup variants)...")
    loader = LangGapLoader(args.langgap_dir, scene_ids=[0, 1, 2, 3])
    samples = loader.load()

    # ── 2. Cross-pair ──────────────────────────────────────────────────────
    print("\nCreating cross-paired samples (image[task i] × instruction[task i+1])...")
    cross_samples, image_labels, instruction_labels = make_cross_pairs(samples)
    print(f"  {len(cross_samples)} cross-paired samples")
    unique_img, counts_img = np.unique(image_labels, return_counts=True)
    unique_ins, counts_ins = np.unique(instruction_labels, return_counts=True)
    print(f"  image labels:       {dict(zip(unique_img.tolist(), counts_img.tolist()))}")
    print(f"  instruction labels: {dict(zip(unique_ins.tolist(), counts_ins.tolist()))}")

    # ── 3. Extract hidden states (once) ───────────────────────────────────
    print("\nExtracting PaliGemma hidden states on cross-paired inputs...")
    extractor = VLMExtractor(args.model, device=args.device)
    raw_layer_data = extractor.extract(cross_samples)

    # ── 4. Build two layer_data dicts with different labels ───────────────
    visual_layer_data = {
        layer: (X, image_labels)
        for layer, (X, _) in raw_layer_data.items()
    }
    language_layer_data = {
        layer: (X, instruction_labels)
        for layer, (X, _) in raw_layer_data.items()
    }

    # ── 5. Train probes ────────────────────────────────────────────────────
    print("\nTraining visual probe (label = image's task)...")
    visual_results = train_probes(
        visual_layer_data, cv_folds=args.cv_folds, verbose=True
    )
    save_results(visual_results, out / "cross_visual_results.json", component="vlm")

    print("\nTraining language probe (label = instruction's task)...")
    language_results = train_probes(
        language_layer_data, cv_folds=args.cv_folds, verbose=True
    )
    save_results(language_results, out / "cross_language_results.json", component="vlm")

    # ── 6. Plot + summary ─────────────────────────────────────────────────
    print_cross_summary(visual_results, language_results)
    plot_cross_probe(
        visual_results,
        language_results,
        save_path=out / "cross_probe_accuracy.png",
        show=not args.no_show,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="google/paligemma-3b-pt-224")
    p.add_argument("--langgap_dir", default="./langgap_hf")
    p.add_argument("--output_dir",  default="./results/cross")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--cv_folds",    type=int, default=5)
    p.add_argument("--no_show",     action="store_true")
    main(p.parse_args())

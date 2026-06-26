"""
probes/evaluate.py
==================
Loads saved probe results and generates the key diagnostic plot:
    probe accuracy vs layer index for VLM and/or action expert.

How to read the plot:
    - X axis: layer index (early = left, late = right)
    - Y axis: cross-validated logistic regression accuracy
    - Dashed horizontal line: chance level (1/n_classes)
    - Purple curve: VLM backbone layers
    - Coral curve:  action expert layers

Interpretation:
    VLM curve rises → VLM encodes object identity in its hidden states.
    Action expert curve stays flat → language gap confirmed.
    Gap between curves → exactly the information lost in transfer.
"""

from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .train import load_results


matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

VLM_COLOR = "#534AB7"      # purple
ACTION_COLOR = "#D85A30"   # coral


def plot_probe_accuracy(
    vlm_results_path: Optional[str | Path] = None,
    action_results_path: Optional[str | Path] = None,
    save_path: Optional[str | Path] = None,
    title: str = "Object identity probe accuracy by layer",
    show: bool = True,
) -> plt.Figure:
    """
    Load results from JSON files and produce the overlay plot.

    Args:
        vlm_results_path:    path to VLM probe results .json (or None)
        action_results_path: path to action expert results .json (or None)
        save_path:           where to save the PNG (None = don't save)
        title:               plot title
        show:                call plt.show() after plotting
    """
    fig, ax = plt.subplots(figsize=(11, 5))

    n_classes = 3  # updated if we load real results
    curves_plotted = 0

    if vlm_results_path is not None:
        _, vlm_res = load_results(vlm_results_path)
        if vlm_res:
            n_classes = list(vlm_res.values())[0]["n_classes"]
            layers = sorted(vlm_res)
            accs = [vlm_res[l]["accuracy"] for l in layers]
            ax.plot(
                layers, accs,
                "o-", color=VLM_COLOR, linewidth=2.2, markersize=4.5,
                label="VLM backbone layers",
            )
            curves_plotted += 1

    if action_results_path is not None:
        _, ae_res = load_results(action_results_path)
        if ae_res:
            n_classes = list(ae_res.values())[0]["n_classes"]
            layers = sorted(ae_res)
            accs = [ae_res[l]["accuracy"] for l in layers]
            ax.plot(
                layers, accs,
                "s--", color=ACTION_COLOR, linewidth=2.2, markersize=4.5,
                label="Action expert layers",
            )
            curves_plotted += 1

    if curves_plotted == 0:
        raise ValueError("No results loaded. Provide at least one results path.")

    chance = 1.0 / n_classes
    ax.axhline(
        chance, color="#888780", linestyle=":", linewidth=1.2,
        label=f"Chance ({n_classes} classes = {chance:.2f})",
    )

    ax.set_xlabel("Layer index", fontsize=12)
    ax.set_ylabel("CV probe accuracy", fontsize=12)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=11)
    ax.tick_params(labelsize=11)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {save_path}")

    if show:
        plt.show()

    return fig


def print_summary(
    vlm_results_path: Optional[str | Path] = None,
    action_results_path: Optional[str | Path] = None,
):
    """Print a text summary of what the probe results mean."""
    print("\n" + "=" * 60)
    print("PROBE RESULTS SUMMARY")
    print("=" * 60)

    if vlm_results_path:
        _, vlm_res = load_results(vlm_results_path)
        chance = list(vlm_res.values())[0]["chance"]
        max_layer = max(vlm_res, key=lambda l: vlm_res[l]["accuracy"])
        max_acc = vlm_res[max_layer]["accuracy"]
        above_chance_layers = [l for l, v in vlm_res.items() if v["above_chance"]]
        first_signal = min(above_chance_layers) if above_chance_layers else None

        print(f"\nVLM backbone:")
        print(f"  Peak accuracy: {max_acc:.3f} at layer {max_layer}  (chance={chance:.3f})")
        if first_signal is not None:
            print(f"  Object identity first encodeable at layer {first_signal}")
            print(f"  → VLM DOES encode which object is named ✓")
        else:
            print(f"  No layer exceeds chance + 0.10 threshold")
            print(f"  → VLM does NOT encode object identity (unexpected)")

    if action_results_path:
        _, ae_res = load_results(action_results_path)
        chance = list(ae_res.values())[0]["chance"]
        max_layer = max(ae_res, key=lambda l: ae_res[l]["accuracy"])
        max_acc = ae_res[max_layer]["accuracy"]
        above_chance_layers = [l for l, v in ae_res.items() if v["above_chance"]]

        print(f"\nAction expert:")
        print(f"  Peak accuracy: {max_acc:.3f} at layer {max_layer}  (chance={chance:.3f})")
        if above_chance_layers:
            print(f"  Above-chance layers: {above_chance_layers}")
            print(f"  → Action expert DOES encode some object identity ✓")
        else:
            print(f"  No layer exceeds chance + 0.10 threshold")
            print(f"  → Language gap confirmed: action expert is object-blind ✗")

    if vlm_results_path and action_results_path:
        print(f"\nDiagnosis:")
        _, vlm_res = load_results(vlm_results_path)
        _, ae_res  = load_results(action_results_path)
        vlm_peak = max(v["accuracy"] for v in vlm_res.values())
        ae_peak  = max(v["accuracy"] for v in ae_res.values())
        gap = vlm_peak - ae_peak
        if gap > 0.2:
            print(f"  Large transfer gap ({gap:.3f}): "
                  "object info exists in VLM but not reaching action expert")
        elif gap > 0.05:
            print(f"  Moderate transfer gap ({gap:.3f}): "
                  "partial language use by action expert")
        else:
            print(f"  Small gap ({gap:.3f}): "
                  "action expert appears to encode object identity")

    print("=" * 60 + "\n")

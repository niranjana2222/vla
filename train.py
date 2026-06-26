"""
probes/train.py
===============
Trains one logistic regression probe per layer on extracted hidden states.

Why logistic regression?
    It's a *linear* classifier — it can only find linear separating boundaries
    in the hidden state space. If it achieves high accuracy at layer L, that
    means object identity is *linearly decodable* from layer L's representation.
    This is a strong interpretability result: it means the information is in a
    direction that could easily be read off by a downstream module.

    A non-linear classifier (e.g. MLP) might find non-linear patterns that
    wouldn't be accessible to a linear cross-attention layer. We use logistic
    regression specifically because it matches the kind of readout that's
    plausible in transformer cross-attention.

Why cross-validation?
    Robotics datasets are small. With 90 samples (30 per class), a simple
    train/test split is noisy. LogisticRegressionCV does k-fold CV internally
    and also selects the regularisation strength C automatically.

Why StandardScaler?
    Different layers have very different activation magnitude ranges. Layer 1
    vectors might be in [-0.1, 0.1], layer 28 in [-10, 10]. The logistic
    regression solver is sensitive to scale, so we normalise each layer's
    hidden states to zero mean and unit variance independently.
"""

import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_probes(
    layer_data: dict[int, tuple[np.ndarray, np.ndarray]],
    cv_folds: int = 5,
    max_iter: int = 1000,
    random_state: int = 42,
    verbose: bool = True,
) -> dict[int, dict]:
    """
    Fit one probe per layer.

    Args:
        layer_data:    {layer_idx: (X, y)} from an extractor
        cv_folds:      number of cross-validation folds
        max_iter:      max solver iterations (increase if convergence warnings)
        random_state:  reproducibility seed
        verbose:       print layer-by-layer accuracy

    Returns:
        {layer_idx: {
            "accuracy":  float,   ← mean CV accuracy across folds at best C
            "n_samples": int,
            "n_classes": int,
            "chance":    float,   ← 1 / n_classes
            "above_chance": bool, ← accuracy > chance + 0.10
        }}
        Note: the fitted sklearn Pipeline is NOT stored here (can't be JSON-serialised).
        Use save_results() to persist the numeric results.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    results = {}

    for layer_idx, (X, y) in sorted(layer_data.items()):
        n_classes = len(np.unique(y))
        chance = 1.0 / n_classes

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegressionCV(
                cv=skf,
                max_iter=max_iter,
                class_weight="balanced",
                solver="saga",       # handles large hidden_dim well
                random_state=random_state,
                n_jobs=-1,
            )),
        ])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X, y)

        # scores_ shape: {class: (n_folds, n_C)}  in OvR, or just one key
        clf = pipe.named_steps["clf"]
        scores_arr = list(clf.scores_.values())[0]
        best_c_idx = np.argmax(scores_arr.mean(axis=0))
        cv_acc = float(scores_arr.mean(axis=0)[best_c_idx])

        results[layer_idx] = {
            "accuracy": cv_acc,
            "n_samples": len(y),
            "n_classes": n_classes,
            "chance": chance,
            "above_chance": cv_acc > chance + 0.10,
        }

        if verbose:
            marker = " ✓" if results[layer_idx]["above_chance"] else ""
            print(
                f"  Layer {layer_idx:3d}  "
                f"acc={cv_acc:.3f}  "
                f"(chance={chance:.3f}){marker}"
            )

    return results


def save_results(
    results: dict[int, dict],
    path: str | Path,
    component: str = "vlm",
):
    """Save probe results to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "component": component,
        "layers": {str(k): v for k, v in results.items()},
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved → {path}")


def load_results(path: str | Path) -> tuple[str, dict[int, dict]]:
    """Load probe results. Returns (component_name, {layer_idx: dict})."""
    with open(path) as f:
        payload = json.load(f)
    component = payload["component"]
    results = {int(k): v for k, v in payload["layers"].items()}
    return component, results

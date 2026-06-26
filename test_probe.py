"""
tests/test_probe.py
===================
Unit tests for probe training and evaluation logic.
All tests use synthetic data — no model loading.

Run:  python -m pytest tests/test_probe.py -v
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from train import train_probes, save_results, load_results


def make_layer_data(
    n_samples=60,
    n_classes=3,
    n_layers=10,
    hidden_dim=64,
    signal_layer=7,
    seed=0,
):
    """Helper: generates {layer: (X, y)} with signal only in late layers."""
    rng = np.random.default_rng(seed)
    y = np.array([i % n_classes for i in range(n_samples)])
    means = rng.normal(0, 3, (n_classes, hidden_dim))
    layer_data = {}
    for l in range(n_layers):
        X = rng.normal(0, 1, (n_samples, hidden_dim)).astype(np.float32)
        if l >= signal_layer:
            for i, label in enumerate(y):
                X[i] += means[label]
        layer_data[l] = (X, y)
    return layer_data, y


class TestTrainProbes:
    def test_returns_all_layers(self):
        data, _ = make_layer_data(n_layers=10)
        results = train_probes(data, cv_folds=3, verbose=False)
        assert set(results.keys()) == set(range(10))

    def test_accuracy_range(self):
        data, _ = make_layer_data()
        results = train_probes(data, cv_folds=3, verbose=False)
        for v in results.values():
            assert 0.0 <= v["accuracy"] <= 1.0

    def test_late_layers_above_chance(self):
        """Late layers should have meaningful signal."""
        data, _ = make_layer_data(n_samples=90, signal_layer=7, n_layers=10)
        results = train_probes(data, cv_folds=3, verbose=False)
        late_accs = [results[l]["accuracy"] for l in range(8, 10)]
        assert max(late_accs) > 0.5, (
            f"Expected late layers to exceed 0.5, got {max(late_accs):.3f}"
        )

    def test_early_layers_near_chance(self):
        """Early layers should be near chance (no signal injected)."""
        data, _ = make_layer_data(n_samples=90, signal_layer=7, n_layers=10)
        results = train_probes(data, cv_folds=3, verbose=False)
        chance = 1 / 3
        early_acc = results[0]["accuracy"]
        assert early_acc < chance + 0.20, (
            f"Expected early layer near chance ({chance:.3f}), got {early_acc:.3f}"
        )

    def test_n_classes_reported_correctly(self):
        data, _ = make_layer_data(n_classes=4)
        results = train_probes(data, cv_folds=3, verbose=False)
        for v in results.values():
            assert v["n_classes"] == 4

    def test_chance_is_correct_fraction(self):
        for n_classes in [2, 3, 4]:
            data, _ = make_layer_data(n_classes=n_classes)
            results = train_probes(data, cv_folds=3, verbose=False)
            for v in results.values():
                assert abs(v["chance"] - 1 / n_classes) < 1e-6

    def test_above_chance_flag(self):
        data, _ = make_layer_data(n_samples=90, signal_layer=7, n_layers=10)
        results = train_probes(data, cv_folds=3, verbose=False)
        for l, v in results.items():
            expected = v["accuracy"] > v["chance"] + 0.10
            assert v["above_chance"] == expected

    def test_n_samples_reported(self):
        n = 75
        data, _ = make_layer_data(n_samples=n, n_layers=3)
        results = train_probes(data, cv_folds=3, verbose=False)
        for v in results.values():
            assert v["n_samples"] == n


class TestSaveLoadResults:
    def test_round_trip(self):
        data, _ = make_layer_data(n_layers=5)
        results = train_probes(data, cv_folds=3, verbose=False)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_results.json"
            save_results(results, path, component="vlm")

            assert path.exists()
            component, loaded = load_results(path)

        assert component == "vlm"
        assert set(loaded.keys()) == set(results.keys())
        for l in results:
            assert abs(loaded[l]["accuracy"] - results[l]["accuracy"]) < 1e-9

    def test_creates_parent_dirs(self):
        data, _ = make_layer_data(n_layers=3)
        results = train_probes(data, cv_folds=3, verbose=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "dir" / "results.json"
            save_results(results, path)
            assert path.exists()

    def test_json_is_readable(self):
        data, _ = make_layer_data(n_layers=3)
        results = train_probes(data, cv_folds=3, verbose=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.json"
            save_results(results, path)
            with open(path) as f:
                raw = json.load(f)
        assert "component" in raw
        assert "layers" in raw
        assert all(isinstance(k, str) for k in raw["layers"])

    def test_component_name_preserved(self):
        data, _ = make_layer_data(n_layers=3)
        results = train_probes(data, cv_folds=3, verbose=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.json"
            save_results(results, path, component="action_expert")
            comp, _ = load_results(path)
        assert comp == "action_expert"

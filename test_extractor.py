"""
tests/test_extractor.py
========================
Tests for the extraction pipeline using SyntheticExtractor.
No real model loading.

Run:  python -m pytest tests/test_extractor.py -v
"""

import numpy as np
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.synthetic import SyntheticLoader, SyntheticExtractor


@pytest.fixture
def samples():
    return SyntheticLoader(n_per_class=20, seed=0).load()


@pytest.fixture
def extractor():
    return SyntheticExtractor(
        n_vlm_layers=10,
        n_ae_layers=6,
        hidden_dim=64,
        seed=0,
    )


class TestSyntheticExtractor:
    def test_vlm_returns_all_layers(self, samples, extractor):
        data = extractor.extract_vlm(samples)
        assert set(data.keys()) == set(range(10))

    def test_ae_returns_all_layers(self, samples, extractor):
        data = extractor.extract_action_expert(samples)
        assert set(data.keys()) == set(range(6))

    def test_vlm_x_shape(self, samples, extractor):
        data = extractor.extract_vlm(samples)
        for layer, (X, y) in data.items():
            assert X.shape == (len(samples), 64), f"Layer {layer}: wrong shape {X.shape}"
            assert y.shape == (len(samples),)

    def test_vlm_dtype(self, samples, extractor):
        data = extractor.extract_vlm(samples)
        for X, y in data.values():
            assert X.dtype == np.float32
            assert y.dtype in [np.int32, np.int64, int]

    def test_labels_match_samples(self, samples, extractor):
        data = extractor.extract_vlm(samples)
        expected_labels = [s["label"] for s in samples]
        for X, y in data.values():
            np.testing.assert_array_equal(y, expected_labels)

    def test_vlm_late_layers_have_signal(self, samples, extractor):
        """Late VLM layers should be above chance after probing."""
        from probes.train import train_probes
        data = extractor.extract_vlm(samples)
        results = train_probes(data, cv_folds=3, verbose=False)
        late_acc = max(results[l]["accuracy"] for l in range(8, 10))
        assert late_acc > 0.45, f"Expected late VLM signal, got {late_acc:.3f}"

    def test_ae_layers_near_chance(self, samples, extractor):
        """Action expert should stay near chance (no signal)."""
        from probes.train import train_probes
        data = extractor.extract_action_expert(samples)
        results = train_probes(data, cv_folds=3, verbose=False)
        chance = 1 / 3
        peak = max(v["accuracy"] for v in results.values())
        assert peak < chance + 0.25, (
            f"Action expert should stay near chance, got peak={peak:.3f}"
        )

    def test_vlm_no_nans(self, samples, extractor):
        data = extractor.extract_vlm(samples)
        for layer, (X, _) in data.items():
            assert not np.isnan(X).any(), f"NaN in VLM layer {layer}"

    def test_ae_no_nans(self, samples, extractor):
        data = extractor.extract_action_expert(samples)
        for layer, (X, _) in data.items():
            assert not np.isnan(X).any(), f"NaN in AE layer {layer}"

    def test_different_seeds_different_vectors(self):
        loader = SyntheticLoader(n_per_class=10, seed=0)
        samples = loader.load()
        e1 = SyntheticExtractor(n_vlm_layers=3, hidden_dim=32, seed=0)
        e2 = SyntheticExtractor(n_vlm_layers=3, hidden_dim=32, seed=99)
        d1 = e1.extract_vlm(samples)
        d2 = e2.extract_vlm(samples)
        # Different seeds should produce different vectors
        assert not np.allclose(d1[0][0], d2[0][0])

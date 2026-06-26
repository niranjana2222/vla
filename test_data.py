"""
tests/test_data.py
==================
Tests for data loaders (SyntheticLoader + LangGapLoader fallback behaviour).

Run:  python -m pytest tests/test_data.py -v
"""

import numpy as np
import pytest
from pathlib import Path
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from synthetic import SyntheticLoader
from langgap_loader import LangGapLoader


class TestSyntheticLoader:
    def test_correct_sample_count(self):
        loader = SyntheticLoader(n_per_class=30)
        samples = loader.load()
        assert len(samples) == 90  # 3 classes × 30

    def test_sample_keys(self):
        samples = SyntheticLoader(n_per_class=5).load()
        for s in samples:
            assert "image" in s
            assert "instruction" in s
            assert "label" in s

    def test_images_are_pil(self):
        samples = SyntheticLoader(n_per_class=5).load()
        for s in samples:
            assert isinstance(s["image"], Image.Image)

    def test_image_size(self):
        samples = SyntheticLoader(n_per_class=5, image_size=64).load()
        for s in samples:
            assert s["image"].size == (64, 64)

    def test_labels_correct_range(self):
        loader = SyntheticLoader(n_per_class=10)
        samples = loader.load()
        labels = [s["label"] for s in samples]
        assert set(labels) == {0, 1, 2}

    def test_n_classes(self):
        loader = SyntheticLoader()
        assert loader.n_classes == 3

    def test_class_names_length(self):
        loader = SyntheticLoader()
        assert len(loader.class_names) == 3

    def test_balanced_classes(self):
        n = 25
        loader = SyntheticLoader(n_per_class=n)
        samples = loader.load()
        from collections import Counter
        counts = Counter(s["label"] for s in samples)
        assert all(c == n for c in counts.values())

    def test_instructions_are_strings(self):
        samples = SyntheticLoader(n_per_class=5).load()
        for s in samples:
            assert isinstance(s["instruction"], str)
            assert len(s["instruction"]) > 10

    def test_different_seeds_different_images(self):
        s1 = SyntheticLoader(n_per_class=5, seed=0).load()
        s2 = SyntheticLoader(n_per_class=5, seed=99).load()
        arr1 = np.array(s1[0]["image"])
        arr2 = np.array(s2[0]["image"])
        assert not np.array_equal(arr1, arr2)

    def test_same_seed_reproducible(self):
        s1 = SyntheticLoader(n_per_class=5, seed=42).load()
        s2 = SyntheticLoader(n_per_class=5, seed=42).load()
        arr1 = np.array(s1[0]["image"])
        arr2 = np.array(s2[0]["image"])
        np.testing.assert_array_equal(arr1, arr2)


class TestLangGapLoaderFallback:
    def test_raises_on_missing_dir(self):
        with pytest.raises(FileNotFoundError):
            loader = LangGapLoader(langgap_dir="/nonexistent/path/xyz")
            loader.load()

    def test_raises_useful_message(self):
        with pytest.raises(FileNotFoundError, match="langgap"):
            loader = LangGapLoader(langgap_dir="/nonexistent/path/xyz")
            loader.load()

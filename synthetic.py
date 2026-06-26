"""
data/synthetic.py
=================
Generates synthetic (image, instruction, label) samples for smoke testing
the full probe pipeline without needing a GPU, model weights, or LangGap.

The synthetic hidden states have a designed signal:
- Early layers: random (no object information)
- Late layers: Gaussian clusters per class (object information present)

This lets you verify the probe correctly finds the signal layer and that
the plotting / saving pipeline all works before running on real data.
"""

import numpy as np
from PIL import Image


class SyntheticLoader:
    """
    Produces fake PIL images + short instruction strings.
    These are only used to test data plumbing — the model never runs on them
    in smoke test mode (SyntheticExtractor generates the hidden states instead).
    """

    INSTRUCTIONS = [
        "Pick up the milk and place it in the basket",
        "Pick up the tomato sauce and place it in the basket",
        "Pick up the jeans and place it in the basket",
    ]

    def __init__(self, n_per_class: int = 30, image_size: int = 224, seed: int = 42):
        self.n_per_class = n_per_class
        self.image_size = image_size
        self.rng = np.random.default_rng(seed)

    def load(self) -> list[dict]:
        samples = []
        for label, instruction in enumerate(self.INSTRUCTIONS):
            for _ in range(self.n_per_class):
                pixel_data = self.rng.integers(0, 256, (self.image_size, self.image_size, 3), dtype=np.uint8)
                image = Image.fromarray(pixel_data)
                samples.append({
                    "image": image,
                    "instruction": instruction,
                    "label": label,
                    "scene_id": "synthetic",
                    "variant": f"class_{label}",
                })
        print(f"Generated {len(samples)} synthetic samples ({self.n_per_class} per class)")
        return samples

    @property
    def n_classes(self) -> int:
        return len(self.INSTRUCTIONS)

    @property
    def class_names(self) -> list[str]:
        return ["milk", "tomato_sauce", "jeans"]


class SyntheticExtractor:
    """
    Generates synthetic hidden states that simulate the language gap:
    - VLM layers: random at first, then increasingly separable by label
    - Action expert layers: always near random (no signal from VLM)

    Used by smoke_test.py to verify probe training and plotting without
    any model weights.
    """

    def __init__(
        self,
        n_vlm_layers: int = 28,
        n_ae_layers: int = 18,
        hidden_dim: int = 256,
        signal_starts_at_fraction: float = 0.6,
        seed: int = 42,
    ):
        self.n_vlm_layers = n_vlm_layers
        self.n_ae_layers = n_ae_layers
        self.hidden_dim = hidden_dim
        self.signal_starts_at = int(signal_starts_at_fraction * n_vlm_layers)
        self.rng = np.random.default_rng(seed)

    def extract_vlm(
        self, samples: list[dict]
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """
        Returns {layer_idx: (X, y)} where X is (n_samples, hidden_dim).
        Early layers are random. Late layers have class-separable clusters.
        """
        n = len(samples)
        labels = np.array([s["label"] for s in samples])
        n_classes = len(np.unique(labels))

        class_means = self.rng.normal(0, 3.0, (n_classes, self.hidden_dim))

        layer_data = {}
        for layer in range(self.n_vlm_layers):
            progress = max(0, (layer - self.signal_starts_at)) / max(
                1, self.n_vlm_layers - self.signal_starts_at
            )
            X = self.rng.normal(0, 1.0, (n, self.hidden_dim)).astype(np.float32)
            for i, label in enumerate(labels):
                X[i] += progress * class_means[label]
            layer_data[layer] = (X, labels)
        return layer_data

    def extract_action_expert(
        self, samples: list[dict]
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """
        Returns {layer_idx: (X, y)} — all near-random (simulates language gap).
        """
        n = len(samples)
        labels = np.array([s["label"] for s in samples])
        layer_data = {}
        for layer in range(self.n_ae_layers):
            X = self.rng.normal(0, 1.0, (n, self.hidden_dim)).astype(np.float32)
            layer_data[layer] = (X, labels)
        return layer_data

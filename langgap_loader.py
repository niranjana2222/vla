"""
data/langgap_loader.py
======================
Loads scenes from the LangGap benchmark into the format the probe pipeline
expects: a list of dicts with keys {image, instruction, label}.

LangGap structure (from https://github.com/YC11Hou/langgap):
    data/
        scene_0000/
            base/
                image.png
                instruction.txt      "Pick up the milk and place it in the basket"
            extended_0/
                image.png            SAME image as base (same scene layout)
                instruction.txt      "Pick up the tomato sauce and place it in the basket"
            extended_1/
                image.png
                instruction.txt      "Pick up the jeans and place it in the basket"
        scene_0001/
            ...

The key property: image is IDENTICAL across base + all extended variants.
Only the instruction (= object name) changes. This forces any probe signal
to come purely from the instruction, not from visual differences.

If langgap_dir doesn't exist or has no scenes, falls back to SyntheticLoader.
"""

import os
import json
from pathlib import Path
from typing import Optional

from PIL import Image


class LangGapLoader:
    """
    Loads LangGap scenes. Each scene produces N samples sharing one image
    but with different object instructions. Label = instruction index within
    the scene (0 = base, 1 = extended_0, 2 = extended_1, ...).

    Args:
        langgap_dir:  path to the LangGap data/ directory
        scene_ids:    list of scene indices to load (None = all)
        max_per_scene: max number of instructions per scene (including base)
        image_size:   resize images to this square size
    """

    def __init__(
        self,
        langgap_dir: str,
        scene_ids: Optional[list] = None,
        max_per_scene: int = 3,
        image_size: int = 224,
    ):
        self.langgap_dir = Path(langgap_dir)
        self.scene_ids = scene_ids
        self.max_per_scene = max_per_scene
        self.image_size = image_size

        self._scenes = self._discover_scenes()
        if not self._scenes:
            raise FileNotFoundError(
                f"No LangGap scenes found in {langgap_dir}.\n"
                "Run: git clone https://github.com/YC11Hou/langgap\n"
                "Or use SyntheticLoader for smoke testing."
            )

    def _discover_scenes(self) -> list:
        """Return sorted list of scene directories."""
        if not self.langgap_dir.exists():
            return []
        scenes = sorted(p for p in self.langgap_dir.iterdir() if p.is_dir())
        if self.scene_ids is not None:
            scenes = [scenes[i] for i in self.scene_ids if i < len(scenes)]
        return scenes

    def _load_image(self, path: Path) -> Image.Image:
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
        return img

    def _load_instruction(self, scene_dir: Path, variant: str) -> Optional[str]:
        """Load instruction text from scene_dir/variant/instruction.txt."""
        txt_path = scene_dir / variant / "instruction.txt"
        if not txt_path.exists():
            return None
        return txt_path.read_text().strip()

    def _load_image_from_variant(self, scene_dir: Path, variant: str) -> Optional[Image.Image]:
        img_path = scene_dir / variant / "image.png"
        if not img_path.exists():
            img_path = scene_dir / variant / "image.jpg"
        if not img_path.exists():
            return None
        return self._load_image(img_path)

    def load(self) -> list[dict]:
        """
        Returns list of samples:
            {
                "image":       PIL.Image,
                "instruction": str,
                "label":       int,   # 0, 1, 2, ... (object index within scene)
                "scene_id":    str,   # for debugging
                "variant":     str,   # "base" | "extended_0" | ...
            }
        """
        samples = []
        for scene_dir in self._scenes:
            variants = ["base"] + [
                f"extended_{i}" for i in range(self.max_per_scene - 1)
            ]

            scene_samples = []
            for label, variant in enumerate(variants):
                image = self._load_image_from_variant(scene_dir, variant)
                instruction = self._load_instruction(scene_dir, variant)
                if image is None or instruction is None:
                    continue
                scene_samples.append({
                    "image": image,
                    "instruction": instruction,
                    "label": label,
                    "scene_id": scene_dir.name,
                    "variant": variant,
                })

            if len(scene_samples) >= 2:
                samples.extend(scene_samples)

        print(f"Loaded {len(samples)} samples from {len(self._scenes)} scenes")
        print(f"  Labels: {sorted(set(s['label'] for s in samples))}")
        print(f"  Example: '{samples[0]['instruction']}'")
        return samples

    @property
    def n_classes(self) -> int:
        return self.max_per_scene

    @property
    def class_names(self) -> list[str]:
        return [f"object_{i}" for i in range(self.max_per_scene)]

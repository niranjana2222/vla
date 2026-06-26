"""
models/action_extractor.py
==========================
Extracts hidden states from the action expert transformer using PyTorch
forward hooks. Works with any architecture — you only need to implement
`_get_layer_list()` for your specific model.

How hooks work:
    A forward hook is a callback that fires AFTER a module's forward()
    completes. We register one hook per transformer layer. When the action
    expert runs a forward pass, each hook captures the output hidden state
    of its layer and stores it in a dict. We then remove the hooks.

    hook_fn(module, input, output):
        output is typically (hidden_state, ...) or just hidden_state
        hidden_state shape: (batch, seq_len, hidden_dim)
        We take [0, -1, :] → last token of batch 0

Key design decisions:
    - Hooks are registered + removed for every sample. This avoids any
      risk of stale captured values across samples.
    - We use torch.no_grad() everywhere to avoid VRAM accumulation.
    - The VLM context is computed separately (via VLMExtractor or a
      combined model forward) and passed into the action expert.
"""

import numpy as np
import torch
from tqdm import tqdm
from typing import Optional


class ActionExpertExtractor:
    """
    Args:
        action_expert_ckpt: path to .pt checkpoint file
        arch:    "smolvla" | "pi0" | "custom"
                 Determines how _get_layer_list() finds the transformer blocks.
        vlm_extractor: a VLMExtractor instance used to get VLM context
        layers:  layer indices to extract (None = all)
        device:  "cuda" | "cpu"
    """

    def __init__(
        self,
        action_expert_ckpt: Optional[str] = None,
        arch: str = "smolvla",
        vlm_extractor=None,
        layers: Optional[list] = None,
        device: str = "cuda",
    ):
        self.action_expert_ckpt = action_expert_ckpt
        self.arch = arch
        self.vlm_extractor = vlm_extractor
        self.layers = layers
        self.device = device
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        if self.action_expert_ckpt is None:
            raise ValueError(
                "action_expert_ckpt not set. "
                "Set it in config.yaml or pass --action_ckpt to the script."
            )
        print(f"Loading action expert from {self.action_expert_ckpt}...")
        self._model = torch.load(self.action_expert_ckpt, map_location=self.device)
        self._model.eval()
        print("Action expert loaded.")

    def _get_layer_list(self, model) -> list:
        """
        Return the list of transformer block modules to hook.
        ADAPT THIS for your specific action expert architecture.
        """
        if self.arch == "smolvla":
            # SmolVLA / LeRobot style
            return list(model.action_expert.transformer.layers)
        elif self.arch == "pi0":
            # π0 DiT-style action expert
            return list(model.action_expert.blocks)
        elif self.arch == "custom":
            # Generic HuggingFace-style
            return list(model.action_expert.model.layers)
        else:
            raise ValueError(
                f"Unknown arch '{self.arch}'. "
                "Set arch='custom' and edit _get_layer_list() for your model."
            )

    def _get_vlm_context(self, model, image, instruction: str) -> torch.Tensor:
        """
        Get VLM context to pass into the action expert.
        Typically the VLM's last hidden state or past_key_values.
        ADAPT THIS for how your combined model passes context.
        """
        if self.vlm_extractor is not None and hasattr(self.vlm_extractor, "_model"):
            vlm = self.vlm_extractor._model
            processor = self.vlm_extractor._processor
            inputs = processor(
                text=instruction, images=image, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                out = vlm(**inputs, use_cache=True, return_dict=True)
            return out.past_key_values
        elif hasattr(model, "encode_context"):
            return model.encode_context(image, instruction)
        else:
            raise NotImplementedError(
                "Override _get_vlm_context() for your model architecture. "
                "It should return whatever your action expert takes as conditioning."
            )

    @torch.no_grad()
    def _extract_one(self, image, instruction: str) -> dict[int, np.ndarray]:
        """Single sample → {layer_idx: hidden_state_vector}."""
        captured: dict[int, np.ndarray] = {}
        hooks = []

        layer_list = self._get_layer_list(self._model)
        target_layers = self.layers if self.layers is not None else list(range(len(layer_list)))

        def make_hook(layer_idx: int):
            def hook_fn(module, inp, output):
                h = output[0] if isinstance(output, tuple) else output
                # h: (batch, seq_len, hidden_dim) → take last token
                captured[layer_idx] = h[0, -1, :].detach().cpu().float().numpy()
            return hook_fn

        for i in target_layers:
            handle = layer_list[i].register_forward_hook(make_hook(i))
            hooks.append(handle)

        try:
            context = self._get_vlm_context(self._model, image, instruction)
            self._model.action_expert(context=context)
        finally:
            for h in hooks:
                h.remove()

        return captured

    def extract(
        self, samples: list[dict]
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """
        Extract action expert hidden states for all samples.

        Returns:
            {layer_idx: (X, y)}
        """
        self._load()
        if self.vlm_extractor is not None:
            self.vlm_extractor._load()

        layer_vecs: dict[int, list] = {}
        all_labels = []

        for sample in tqdm(samples, desc="Extracting action expert states"):
            states = self._extract_one(sample["image"], sample["instruction"])
            all_labels.append(sample["label"])
            for layer_idx, vec in states.items():
                layer_vecs.setdefault(layer_idx, []).append(vec)

        y = np.array(all_labels, dtype=int)
        return {
            layer_idx: (np.stack(vecs, axis=0), y)
            for layer_idx, vecs in layer_vecs.items()
        }

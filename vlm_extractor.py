"""
models/vlm_extractor.py
=======================
Extracts hidden states from PaliGemma (or any HuggingFace VLM) for each
(image, instruction) sample.

What happens inside one call to .extract(samples):
    for each sample:
        1. processor tokenises instruction + encodes image into input_ids
        2. single forward pass with output_hidden_states=True
        3. grab outputs.hidden_states[layer][0, last_text_idx, :]
           → shape (hidden_dim,) float32 numpy array
        4. store in {layer: list_of_vectors}
    return {layer: (X, y)} for all layers

The "last text token" trick:
    PaliGemma's input sequence is:
        [image patches...] [BOS] ["pick"] ["up"] ["the"] ["milk"] [EOS]
    The EOS token is the last one and has attended to everything before it
    via causal self-attention — so its hidden state is the richest single
    summary of the full input.
"""

import numpy as np
import torch
from tqdm import tqdm
from typing import Optional


class VLMExtractor:
    """
    Args:
        model_name:  HuggingFace model ID, e.g. "google/paligemma-3b-pt-224"
        layers:      list of layer indices to extract (None = all)
        device:      "cuda" | "cpu" | "mps"
        batch_size:  number of samples per forward pass (1 is safest for GPU memory)
        dtype:       torch dtype for the model (float16 saves ~2x VRAM)
    """

    def __init__(
        self,
        model_name: str = "google/paligemma-3b-pt-224",
        layers: Optional[list] = None,
        device: str = "cuda",
        batch_size: int = 1,
        dtype: torch.dtype = torch.float16,
    ):
        self.model_name = model_name
        self.layers = layers
        self.device = device
        self.batch_size = batch_size
        self.dtype = dtype
        self._model = None
        self._processor = None

    def _load(self):
        """Lazy-load model so importing this file is always fast."""
        if self._model is not None:
            return
        from transformers import PaliGemmaForConditionalGeneration, AutoProcessor

        print(f"Loading {self.model_name} on {self.device}...")
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=self.device,
        )
        self._model.eval()
        print(f"Model loaded. Hidden dim: {self._model.config.hidden_size}")

    def _find_last_text_idx(self, input_ids: torch.Tensor) -> int:
        """
        PaliGemma input: [image tokens] [text tokens].
        We want the index of the very last token (EOS), which has attended
        to all image patches AND all instruction tokens.
        """
        return input_ids.shape[1] - 1

    @torch.no_grad()
    def _extract_one(self, image, instruction: str) -> dict[int, np.ndarray]:
        """Single (image, instruction) → {layer: hidden_state_vector}."""
        inputs = self._processor(
            text=instruction,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        outputs = self._model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        # outputs.hidden_states: tuple of (1, seq_len, hidden_dim)
        # length = num_layers + 1 (index 0 = embedding layer)
        all_hidden = outputs.hidden_states
        n_layers = len(all_hidden)
        target_layers = self.layers if self.layers is not None else list(range(n_layers))
        last_idx = self._find_last_text_idx(inputs["input_ids"])

        result = {}
        for layer_idx in target_layers:
            h = all_hidden[layer_idx][0, last_idx, :]  # (hidden_dim,)
            result[layer_idx] = h.cpu().float().numpy()
        return result

    def extract(
        self, samples: list[dict]
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """
        Extract hidden states for all samples.

        Args:
            samples: list of {"image": PIL.Image, "instruction": str, "label": int, ...}

        Returns:
            {layer_idx: (X, y)}
                X: float32 array (n_samples, hidden_dim)
                y: int array (n_samples,)
        """
        self._load()

        layer_vecs: dict[int, list] = {}
        all_labels = []

        for sample in tqdm(samples, desc="Extracting VLM states"):
            states = self._extract_one(sample["image"], sample["instruction"])
            all_labels.append(sample["label"])
            for layer_idx, vec in states.items():
                layer_vecs.setdefault(layer_idx, []).append(vec)

        y = np.array(all_labels, dtype=int)
        return {
            layer_idx: (np.stack(vecs, axis=0), y)
            for layer_idx, vecs in layer_vecs.items()
        }

    @property
    def n_layers(self) -> int:
        self._load()
        return self._model.config.num_hidden_layers + 1

    @property
    def hidden_dim(self) -> int:
        self._load()
        return self._model.config.hidden_size

# VLA Linear Probe — Language Gap Analysis

Diagnoses language blindness in Vision-Language-Action models by training
linear probes on PaliGemma (VLM) and action-expert hidden states, using
LangGap-style object-replacement scenes.

---

## What this does

For each (image, instruction) pair from LangGap:

1. Forward-pass through PaliGemma — extract the last text token's hidden
   state at every layer  → shape `(num_layers+1, hidden_dim)`
2. Forward-pass through the action expert — extract layer outputs via hooks
3. At each layer, fit `LogisticRegressionCV` to classify which object the
   instruction refers to (milk / tomato / jeans etc.)
4. Plot accuracy curves for VLM layers vs action expert layers
5. **Gap between curves = the language gap**: VLM encodes object identity,
   action expert doesn't receive or use it

---

## Project layout

```
vla_probe/
├── README.md               ← this file
├── requirements.txt        ← pip install -r requirements.txt
├── config.yaml             ← all experiment settings in one place
│
├── data/
│   ├── langgap_loader.py   ← loads LangGap scenes (real or synthetic)
│   └── synthetic.py        ← generates smoke-test data without any model
│
├── models/
│   ├── vlm_extractor.py    ← PaliGemma hidden-state extraction
│   └── action_extractor.py ← action expert hook-based extraction
│
├── probes/
│   ├── train.py            ← fits probes at every layer, saves results
│   └── evaluate.py         ← loads saved results, prints & plots
│
├── scripts/
│   ├── run_vlm_probe.py    ← end-to-end: extract VLM states → train → plot
│   ├── run_action_probe.py ← end-to-end: extract action states → train → plot
│   ├── run_both.py         ← runs both and overlays curves
│   └── smoke_test.py       ← verifies full pipeline with synthetic data
│
├── tests/
│   ├── test_extractor.py   ← unit tests for hidden state extraction
│   ├── test_probe.py       ← unit tests for probe training logic
│   └── test_data.py        ← unit tests for data loading
│
└── results/                ← auto-created; stores .json + .png outputs
```

---

## Setup

```bash
# 1. Clone LangGap benchmark
git clone https://github.com/YC11Hou/langgap
cd langgap && pip install -e . && cd ..

# 2. Install probe dependencies
pip install -r requirements.txt

# 3. (Optional) download PaliGemma weights
#    Requires HuggingFace account + signing the model agreement at:
#    https://huggingface.co/google/paligemma-3b-pt-224
huggingface-cli login
python -c "from transformers import PaliGemmaForConditionalGeneration; \
           PaliGemmaForConditionalGeneration.from_pretrained('google/paligemma-3b-pt-224')"
```

---

## Quick start — no GPU needed

Run the smoke test first. It uses **synthetic random data** so you can verify
the whole pipeline before touching any real model or downloading any weights.

```bash
python scripts/smoke_test.py
```

Expected output:
```
=== Smoke test (synthetic data, no model needed) ===
Generating 90 synthetic samples (30 per class)...
Extracting synthetic hidden states...
Training probes on 28 VLM layers...
  Layer  0  CV acc = 0.334  (chance=0.333)
  ...
  Layer 26  CV acc = 0.821  ✓
  Layer 27  CV acc = 0.889  ✓
Training probes on 18 action expert layers...
  Layer  0  CV acc = 0.341  (chance=0.333)
  ...
  Layer 17  CV acc = 0.338  (chance=0.333)   ← language gap visible
Plots saved → results/smoke_test_probe_accuracy.png
=== Smoke test passed ===
```

---

## Full pipeline with PaliGemma + LangGap

### Step 1 — extract VLM hidden states

```bash
python scripts/run_vlm_probe.py \
    --model google/paligemma-3b-pt-224 \
    --langgap_dir ./langgap/data \
    --output_dir ./results/vlm \
    --device cuda          # or cpu (slow but works)
```

This saves `results/vlm/hidden_states.npz` and `results/vlm/labels.npy`.

### Step 2 — extract action expert hidden states

Adapt `config.yaml` to point at your action expert checkpoint, then:

```bash
python scripts/run_action_probe.py \
    --config config.yaml \
    --langgap_dir ./langgap/data \
    --output_dir ./results/action \
    --device cuda
```

### Step 3 — train probes and plot

```bash
python scripts/run_both.py \
    --vlm_results   ./results/vlm \
    --action_results ./results/action \
    --output_dir    ./results \
    --cv_folds 5
```

Saves `results/probe_accuracy_overlay.png` — the key diagnostic plot.

### All-in-one (if you have a combined checkpoint)

```bash
python scripts/run_both.py \
    --model google/paligemma-3b-pt-224 \
    --action_ckpt /path/to/action_expert.pt \
    --langgap_dir ./langgap/data \
    --output_dir  ./results \
    --device cuda
```

---

## Running tests

```bash
# All tests
python -m pytest tests/ -v

# Specific test files
python -m pytest tests/test_extractor.py -v
python -m pytest tests/test_probe.py -v
python -m pytest tests/test_data.py -v

# With coverage
python -m pytest tests/ --cov=. --cov-report=term-missing
```

---

## Reading the output plot

```
Probe accuracy
1.0 │                              ┌──── VLM: object identity encoded ✓
    │                         ┌───┘
0.7 │                    ┌───┘
    │               ┌───┘
0.4 │──────────────┘
    │─────────────────────────────  Action expert: stuck at chance ✗
0.33│ (chance level for 3 classes)
    └──────────────────────────────
        early layers          late layers
```

- VLM curve rises → the VLM **does** encode which object
- Action expert curve stays flat → the action expert **does not** receive it
- The gap between them **localises the language blindness**

---

## Config reference (`config.yaml`)

```yaml
model:
  vlm_name: "google/paligemma-3b-pt-224"
  action_expert_ckpt: null          # set this to your checkpoint path
  action_expert_arch: "smolvla"     # smolvla | pi0 | custom

data:
  langgap_dir: "./langgap/data"
  scene_ids: null                   # null = use all scenes
  instructions_per_scene: 3        # how many object variants per scene
  image_size: 224

probe:
  cv_folds: 5
  max_iter: 1000
  random_state: 42
  label_smoothing: false

output:
  results_dir: "./results"
  save_hidden_states: true          # set false to save disk space
  plot_dpi: 150
```

---

## Adapting to a different action expert

Edit `models/action_extractor.py`. The only thing you need to change is
`_get_layer_list()` — return the list of transformer blocks you want to hook:

```python
def _get_layer_list(self, model):
    # SmolVLA
    return model.action_expert.transformer.layers
    # π0 / DiT style
    return model.action_expert.blocks
    # HuggingFace style
    return model.action_expert.model.layers
```

Everything else (hook registration, extraction, cleanup) is handled automatically.

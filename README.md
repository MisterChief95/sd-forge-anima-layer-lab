# Anima Layer Lab for Forge Neo

An installable Forge Neo tab for expanding and merging a 28-block Anima 2B checkpoint with the 40-block Anima 2.9B architecture. It provides one slider per transformer block and can generate A/B/merged previews directly inside the extension tab.

## Features

- Header-only compatibility scan before large weights are loaded.
- Official Anima 28 → 40 interleaved block mapping.
- Forty transformer-block sliders.
- Six optional LLM-adapter block sliders plus shared-tensor controls.
- Aligned weighted merge, hard A/B interweave, and experimental relative-delta modes.
- Presets for delta graft, all A, all B, alternating blocks, and early-A/late-B blends.
- Streaming safetensors writer with low peak RAM use.
- Atomic output publishing: incomplete merges remain temporary and are removed after errors or cancellation.
- In-tab merged preview and controlled A/B/merged comparison using the same prompt and seed.
- A Forge-native gallery with lightbox and arrow-key navigation.
- Constrained, copyable generation details that cannot widen the tab.
- Automatic JSON merge-config export and import from JSON or embedded safetensors metadata.
- Forge's checkpoint, VAE, and text-encoder settings are restored after each preview transaction.

## Installation

Place this repository in Forge Neo's `extensions` directory:

```text
sd-webui-forge-neo/
└── extensions/
    └── sd-forge-anima-layer-lab/
        ├── anima_layer_lab/
        ├── scripts/
        ├── README.md
        └── style.css
```

Restart Forge Neo. A new **Anima Layer Lab** top-level tab will appear.

No extra Python packages are installed. The extension uses Forge's existing PyTorch, safetensors, Gradio, and generation backend.

## Supported inputs

- **A:** full-precision Anima-family safetensors with 28 main transformer blocks.
- **B:** full-precision Anima 2.9B-family safetensors with 40 main transformer blocks.
- FP16, BF16, and FP32 tensors are accepted. BF16 is recommended.

The first release intentionally rejects GGUF, `int8_convrot`, FP8, and other quantized transformer tensors. Weight interpolation in those formats needs format-specific dequantization and requantization.

The VAE and text encoder are not merged. Select `qwen_image_vae.safetensors` and `qwen_3_06b_base.safetensors` in the preview section.

## Quick start

1. Select the old 2B model as **A** and Anima 2.9B as **B**.
2. Click **Inspect compatibility**.
3. Start with **Recommended layer graft**. It keeps A's inherited blocks and takes B's 12 inserted blocks exactly.
4. Adjust individual layers or use the inherited/inserted group controls.
5. Enter an output filename and click **Build merged checkpoint**.
6. Enter a prompt and click **Generate merged preview**.
7. Use **Compare A / B / merged** for a same-seed three-way comparison.

Click a preview to open Forge's normal lightbox. The left/right arrow keys move between comparison images and Escape closes it.

Outputs are stored under:

```text
models/Stable-diffusion/AnimaLayerLab/
```

## Merge modes

### Aligned weighted merge

The 2B model is expanded into the 40-block layout, then every block uses:

```text
output = A_aligned × (1 - slider) + B × slider
```

The **Recommended layer graft** preset sets the 28 inherited positions to A and the 12 inserted positions to B. Because every preset weight is an endpoint, those tensors are copied byte-for-byte rather than numerically interpolated.

### Hard A/B interweave

Sliders below `0.5` select the aligned A block; sliders at or above `0.5` select the B block. No numerical interpolation is performed.

### Relative delta graft (experimental)

The final 2.9B checkpoint does not include its exact pre-training expansion snapshot. For the non-muted tensors of each inserted block, this mode therefore estimates a relative specialization delta using the corresponding source block in the final B model:

```text
output = A_source + slider × (B_inserted - B_source)
```

For the six output projections that the expansion manifest says were initialized to zero, the formula is exact: `output = slider × B_inserted`. Inherited blocks still use ordinary A/B interpolation. This mode can be useful for experiments, but it is deliberately not the default and does not claim to reproduce B at slider `1`.

## Preview behavior

Forge owns one global inference model. The extension therefore loads the selected preview checkpoint internally, generates through Forge's normal queued processing path, and restores the prior Forge settings afterward. The main txt2img controls do not need to be changed.

Leaving **Reload the previous Forge model after preview** off is faster. The previous settings are still restored, and Forge reloads that model automatically on the next main-tab generation. Turning it on performs the additional physical RAM/VRAM reload immediately.

For comparisons, a seed of `-1` is resolved once and shared by A, B, and merged so the images remain comparable.

## Safety and storage

- An Anima 2.9B BF16 output is roughly 5.4 GiB (about 5.8 GB).
- The writer checks free space before starting.
- Existing outputs are never replaced unless **Overwrite** is explicitly enabled.
- Output is written to a temporary safetensors file, validated, and then atomically renamed.
- Merge recipes are embedded in safetensors metadata under `anima_layer_lab_recipe`.
- A sibling `<model_name>_merge_config.json` file is written automatically after every successful merge.

## Reusing a merge config

Open **Import a previous merge recipe** and either upload the small JSON config or paste a local path to the JSON/merged safetensors file. Loading it restores both model selections, the merge mode and dtype, all 40 transformer weights, six adapter weights, and both shared-tensor weights. The output filename is restored but overwrite remains off as a safety measure.

## Development tests

From the repository root:

```text
python -m unittest discover -v
```

The tests use tiny synthetic 28/40-block checkpoints and verify all-A expansion, all-B identity, relative-delta behavior, header parsing, JSON/metadata config round-trips, overwrite protection, and partial-file cleanup after cancellation.

## Model license

This repository's source code is MIT licensed. Merged model weights remain derivatives of their source checkpoints and retain the applicable CircleStone Labs/NVIDIA model licenses. Check the licenses of both selected inputs before distributing a merge.

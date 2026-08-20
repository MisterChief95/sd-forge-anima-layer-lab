from __future__ import annotations

import html

import gradio as gr

from .config_io import read_merge_config, write_merge_config
from .constants import (
    LAYER_ORIGINS,
    MERGE_MODE_DELTA,
    MERGE_MODE_INTERWEAVE,
    MERGE_MODE_WEIGHTED,
    MERGE_MODES,
    OUTPUT_DTYPE_MATCH_B,
    OUTPUT_DTYPES,
)
from .forge_bridge import (
    checkpoint_path,
    generate_preview,
    model_output_path,
    refresh_forge_choices,
    register_checkpoint,
    resolve_checkpoint,
    sampler_choices,
)
from .inspection import CompatibilityReport, validate_pair
from .merge import MergeRecipe, build_merge


def _default_checkpoint(choices: tuple[str, ...], *, expanded: bool) -> str | None:
    needles = (
        ("2.9b", "2_9b", "2-9b") if expanded else ("anima-base", "anima_base", "preview3", "anima")
    )
    for needle in needles:
        for choice in choices:
            lowered = choice.lower()
            if needle in lowered and (("2.9" in lowered) == expanded or not expanded):
                return choice
    return choices[0] if choices else None


def _format_gib(value: int) -> str:
    return f"{value / 2**30:.2f} GiB"


def _report_html(report: CompatibilityReport) -> str:
    parts = ["<div class='anima-layer-report'>"]
    if report.valid:
        parts.append("<h3 class='anima-ok'>Compatible 28 → 40 Anima pair</h3>")
    else:
        parts.append("<h3 class='anima-error'>Checkpoint pair is not compatible</h3>")
    if report.old is not None:
        parts.append(
            "<p><b>Old:</b> "
            f"{html.escape(report.old.filename)} — {report.old.block_count} blocks, "
            f"{_format_gib(report.old.file_size)}, {', '.join(report.old.dtypes)}</p>",
        )
    if report.new is not None:
        parts.append(
            "<p><b>New:</b> "
            f"{html.escape(report.new.filename)} — {report.new.block_count} blocks, "
            f"{_format_gib(report.new.file_size)}, {', '.join(report.new.dtypes)}</p>",
        )
    parts.append(
        f"<p>Matched {report.matched_tensor_count} tensors; checked "
        f"{report.checked_block_tensor_count} transformer tensors.</p>",
    )
    if report.errors:
        parts.append("<ul class='anima-errors'>")
        parts.extend(f"<li>{html.escape(message)}</li>" for message in report.errors)
        parts.append("</ul>")
    if report.warnings:
        parts.append("<details><summary>Warnings</summary><ul>")
        parts.extend(f"<li>{html.escape(message)}</li>" for message in report.warnings[:30])
        parts.append("</ul></details>")
    parts.append("</div>")
    return "".join(parts)


def inspect_models_ui(old_choice: str, new_choice: str) -> str:
    try:
        report = validate_pair(checkpoint_path(old_choice), checkpoint_path(new_choice))
        return _report_html(report)
    except Exception as exc:
        return f"<div class='error'>{html.escape(str(exc))}</div>"


def refresh_choices_ui(old_value, new_value, module_values):
    choices = refresh_forge_choices()
    old_value = (
        old_value
        if old_value in choices.checkpoints
        else _default_checkpoint(choices.checkpoints, expanded=False)
    )
    new_value = (
        new_value
        if new_value in choices.checkpoints
        else _default_checkpoint(choices.checkpoints, expanded=True)
    )
    kept_modules = [value for value in (module_values or []) if value in choices.modules]
    if not kept_modules:
        kept_modules = list(choices.default_modules)
    return (
        gr.update(choices=list(choices.checkpoints), value=old_value),
        gr.update(choices=list(choices.checkpoints), value=new_value),
        gr.update(choices=list(choices.modules), value=kept_modules),
    )


def _preset_values(name: str):
    if name == "layer_graft":
        mode = MERGE_MODE_WEIGHTED
        layers = [1.0 if origin.inserted else 0.0 for origin in LAYER_ORIGINS]
        non_block = 0.0
        llm_shared = 0.0
        adapters = [0.0] * 6
    elif name == "delta":
        mode = MERGE_MODE_DELTA
        layers = [1.0 if origin.inserted else 0.0 for origin in LAYER_ORIGINS]
        non_block = 0.0
        llm_shared = 0.0
        adapters = [0.0] * 6
    elif name == "all_a":
        mode = MERGE_MODE_WEIGHTED
        layers = [0.0] * 40
        non_block = 0.0
        llm_shared = 0.0
        adapters = [0.0] * 6
    elif name == "all_b":
        mode = MERGE_MODE_WEIGHTED
        layers = [1.0] * 40
        non_block = 1.0
        llm_shared = 1.0
        adapters = [1.0] * 6
    elif name == "alternating":
        mode = MERGE_MODE_INTERWEAVE
        layers = [float(index % 2) for index in range(40)]
        non_block = 0.0
        llm_shared = 0.0
        adapters = [0.0] * 6
    elif name == "early_a_late_b":
        mode = MERGE_MODE_WEIGHTED
        layers = [min(1.0, max(0.0, (index - 12) / 15.0)) for index in range(40)]
        non_block = 0.0
        llm_shared = 0.0
        adapters = [0.0] * 6
    else:
        raise ValueError(name)
    return [mode, non_block, llm_shared, *layers, *adapters]


def _apply_group_weights(original_weight: float, inserted_weight: float):
    return [inserted_weight if origin.inserted else original_weight for origin in LAYER_ORIGINS]


def _uploaded_path(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        path = value.get("path") or value.get("name")
        if path:
            return str(path)
    path = getattr(value, "name", None)
    if path:
        return str(path)
    raise ValueError("Select a merge config or merged safetensors file")


def load_merge_config_ui(value, local_path: str):
    source = (local_path or "").strip()
    if not source:
        source = _uploaded_path(value)
    recipe = read_merge_config(source)
    old_name = resolve_checkpoint(str(recipe["old_path"])).name
    new_name = resolve_checkpoint(str(recipe["new_path"])).name
    output_name = str(recipe["output_path"])
    status = (
        "<div class='anima-layer-report'><h3 class='anima-ok'>Config loaded</h3>"
        "<p>Models, merge mode, dtype, and all 48 weight controls were restored.</p>"
        "</div>"
    )
    return [
        old_name,
        new_name,
        output_name,
        recipe["mode"],
        recipe["output_dtype"],
        False,
        recipe["non_block_weight"],
        recipe["llm_shared_weight"],
        *recipe["layer_weights"],
        *recipe["adapter_weights"],
        status,
    ]


def build_merge_ui(
    old_choice: str,
    new_choice: str,
    filename: str,
    mode: str,
    output_dtype: str,
    overwrite: bool,
    non_block_weight: float,
    llm_shared_weight: float,
    *weights,
):
    from modules import shared

    layer_weights = tuple(float(value) for value in weights[:40])
    adapter_weights = tuple(float(value) for value in weights[40:46])
    output_path = model_output_path(filename)
    recipe = MergeRecipe(
        old_path=checkpoint_path(old_choice),
        new_path=checkpoint_path(new_choice),
        output_path=output_path,
        mode=mode,
        layer_weights=layer_weights,
        adapter_weights=adapter_weights,
        non_block_weight=float(non_block_weight),
        llm_shared_weight=float(llm_shared_weight),
        output_dtype=output_dtype,
        overwrite=bool(overwrite),
    )

    def progress(done: int, total: int, key: str):
        shared.state.sampling_steps = total
        shared.state.sampling_step = done
        shared.state.textinfo = f"Merging tensor {done + 1}/{total}: {key}"
        return not shared.state.interrupted

    result = build_merge(recipe, progress=progress)
    config_path = write_merge_config(recipe)
    registered_name = register_checkpoint(result.output_path)
    status = (
        "<div class='anima-build-success'><h3>Merge ready</h3>"
        f"<p><b>{html.escape(registered_name)}</b><br>"
        f"{html.escape(result.output_path)}<br>"
        f"{_format_gib(result.file_size)} · {result.tensor_count} tensors<br>"
        f"Config: {html.escape(config_path)}</p></div>"
    )
    return registered_name, registered_name, config_path, status


def _preview_common_args(values):
    (
        modules,
        prompt,
        negative_prompt,
        seed,
        width,
        height,
        steps,
        cfg,
        shift,
        sampler,
        scheduler,
        save_preview,
        restore_loaded_model,
    ) = values
    if not modules:
        raise ValueError("Select the Qwen3 0.6B text encoder and Qwen Image VAE")
    return {
        "modules": list(modules),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": int(seed),
        "width": int(width),
        "height": int(height),
        "steps": int(steps),
        "cfg": float(cfg),
        "shift": float(shift),
        "sampler": sampler,
        "scheduler": scheduler,
        "save_preview": bool(save_preview),
        "restore_loaded_model": bool(restore_loaded_model),
    }


def preview_merged_ui(built_checkpoint: str, *values):
    if not built_checkpoint:
        raise ValueError("Build a merged checkpoint first")
    kwargs = _preview_common_args(values)
    return generate_preview([("Merged", built_checkpoint)], **kwargs)


def compare_ui(old_choice: str, new_choice: str, built_checkpoint: str, *values):
    if not built_checkpoint:
        raise ValueError("Build a merged checkpoint first")
    kwargs = _preview_common_args(values)
    return generate_preview(
        [
            ("A · Old 2B", old_choice),
            ("B · New 2.9B", new_choice),
            ("Merged", built_checkpoint),
        ],
        **kwargs,
    )


def create_ui():
    from modules import call_queue, shared

    try:
        choices = refresh_forge_choices()
    except Exception:
        choices = None
    checkpoint_choices = list(choices.checkpoints) if choices else []
    module_choices = list(choices.modules) if choices else []
    default_modules = list(choices.default_modules) if choices else []
    samplers, schedulers = sampler_choices()
    default_sampler = "Euler" if "Euler" in samplers else samplers[0]
    default_scheduler = next(
        (item for item in schedulers if item.lower() == "sgm uniform"),
        schedulers[0],
    )

    with gr.Blocks(analytics_enabled=False, elem_id="anima_layer_lab") as interface:
        gr.Markdown(
            """
## Anima Layer Lab

Expand a 28-block Anima 2B checkpoint into the 40-block Anima 2.9B layout, merge it per layer, and generate controlled previews without leaving this tab.
""",
        )
        built_checkpoint = gr.State("")

        with gr.Row(equal_height=False):
            with gr.Column(scale=4, variant="panel"):
                gr.Markdown("### 1 · Models and merge recipe")
                with gr.Row():
                    old_model = gr.Dropdown(
                        label="Old Anima 2B (A · 28 blocks)",
                        choices=checkpoint_choices,
                        value=_default_checkpoint(tuple(checkpoint_choices), expanded=False),
                    )
                    new_model = gr.Dropdown(
                        label="Anima 2.9B (B · 40 blocks)",
                        choices=checkpoint_choices,
                        value=_default_checkpoint(tuple(checkpoint_choices), expanded=True),
                    )
                with gr.Row():
                    refresh_button = gr.Button("Refresh model lists")
                    inspect_button = gr.Button("Inspect compatibility", variant="secondary")
                compatibility = gr.HTML("Select both checkpoints, then inspect them.")

                with gr.Accordion("Import a previous merge recipe", open=False):
                    merge_config_upload = gr.File(
                        label="Merge config JSON",
                        file_types=[".json"],
                        type="filepath",
                    )
                    merge_config_path = gr.Textbox(
                        label="Or local JSON / merged safetensors path",
                        placeholder="Paste a local path to read embedded metadata without uploading a large checkpoint",
                    )
                    load_config_button = gr.Button("Load merge config")
                    config_status = gr.HTML("")

                mode = gr.Dropdown(
                    label="Merge mode",
                    choices=list(MERGE_MODES),
                    value=MERGE_MODE_WEIGHTED,
                )
                with gr.Row():
                    preset_layer_graft = gr.Button("Recommended layer graft")
                    preset_all_a = gr.Button("All A")
                    preset_all_b = gr.Button("All B")
                with gr.Row():
                    preset_delta = gr.Button("Experimental relative delta")
                    preset_alternating = gr.Button("Alternating A/B")
                    preset_early_late = gr.Button("Early A → late B")

                with gr.Accordion("Shared and LLM-adapter weights", open=False):
                    non_block_weight = gr.Slider(
                        0,
                        1,
                        value=0,
                        step=0.01,
                        label="Non-transformer tensors · A ↔ B",
                    )
                    llm_shared_weight = gr.Slider(
                        0,
                        1,
                        value=0,
                        step=0.01,
                        label="LLM adapter shared tensors · A ↔ B",
                    )
                    adapter_sliders = [
                        gr.Slider(
                            0,
                            1,
                            value=0,
                            step=0.01,
                            label=f"LLM adapter block {index} · A ↔ B",
                        )
                        for index in range(6)
                    ]

                with gr.Row():
                    original_group = gr.Slider(
                        0,
                        1,
                        value=0,
                        step=0.01,
                        label="Set all inherited blocks",
                    )
                    inserted_group = gr.Slider(
                        0,
                        1,
                        value=1,
                        step=0.01,
                        label="Set all inserted blocks",
                    )
                    apply_groups = gr.Button("Apply groups")

                layer_sliders = []
                for start in range(0, 40, 10):
                    with gr.Accordion(
                        f"Transformer blocks {start:02d}–{start + 9:02d}",
                        open=(start == 0),
                    ):
                        for row_start in range(start, start + 10, 2):
                            with gr.Row():
                                for index in range(row_start, row_start + 2):
                                    origin = LAYER_ORIGINS[index]
                                    if origin.inserted:
                                        label = f"Block {index:02d} · NEW from old {origin.source_old_index:02d}"
                                        value = 1.0
                                        classes = ["anima-inserted-layer"]
                                    else:
                                        label = f"Block {index:02d} · old {origin.old_index:02d}"
                                        value = 0.0
                                        classes = ["anima-original-layer"]
                                    layer_sliders.append(
                                        gr.Slider(
                                            0,
                                            1,
                                            value=value,
                                            step=0.01,
                                            label=label,
                                            elem_classes=classes,
                                        ),
                                    )

                with gr.Row():
                    output_name = gr.Textbox(
                        label="Output filename",
                        value="anima-2.9b-layer-graft.safetensors",
                        scale=3,
                    )
                    output_dtype = gr.Dropdown(
                        label="Output dtype",
                        choices=list(OUTPUT_DTYPES),
                        value=OUTPUT_DTYPE_MATCH_B,
                        scale=1,
                    )
                overwrite = gr.Checkbox(
                    False,
                    label="Overwrite an existing file with the same name",
                )
                build_button = gr.Button("Build merged checkpoint", variant="primary")
                built_display = gr.Textbox(label="Built preview checkpoint", interactive=False)
                generated_config = gr.File(label="Generated merge config", interactive=False)
                build_status = gr.HTML("No merge has been built in this session.")

            with gr.Column(scale=5, variant="panel"):
                gr.Markdown("### 2 · In-tab preview")
                modules = gr.Dropdown(
                    label="Anima VAE and text encoder",
                    choices=module_choices,
                    value=default_modules,
                    multiselect=True,
                    info="Select qwen_3_06b_base and qwen_image_vae.",
                )
                prompt = gr.Textbox(
                    label="Prompt",
                    lines=4,
                    value="masterpiece, best quality, highres, absurdres, safe, 1girl, detailed background",
                )
                negative_prompt = gr.Textbox(
                    label="Negative prompt",
                    lines=2,
                    value="worst quality, low quality, blurry, jpeg artifacts, chromatic aberration",
                )
                with gr.Row():
                    seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
                    width = gr.Slider(512, 1536, value=832, step=64, label="Width")
                    height = gr.Slider(512, 1536, value=1216, step=64, label="Height")
                with gr.Row():
                    sampler = gr.Dropdown(label="Sampler", choices=samplers, value=default_sampler)
                    scheduler = gr.Dropdown(
                        label="Scheduler",
                        choices=schedulers,
                        value=default_scheduler,
                    )
                    steps = gr.Slider(1, 100, value=32, step=1, label="Steps")
                with gr.Row():
                    cfg = gr.Slider(1, 10, value=4, step=0.1, label="CFG")
                    shift = gr.Slider(0, 12, value=3, step=0.1, label="Shift")
                with gr.Row():
                    save_preview = gr.Checkbox(False, label="Save preview images")
                    restore_loaded_model = gr.Checkbox(
                        False,
                        label="Reload the previous Forge model after preview",
                        info="Off is faster; Forge settings are restored either way.",
                    )
                with gr.Row():
                    preview_button = gr.Button("Generate merged preview", variant="primary")
                    compare_button = gr.Button("Compare A / B / merged", variant="secondary")
                gr.Markdown(
                    "Click an image for Forge's lightbox; use **←/→** to navigate and **Esc** to close.",
                )
                with gr.Group(elem_id="anima_layer_lab_gallery_container"):
                    gallery = gr.Gallery(
                        label="Output",
                        show_label=False,
                        elem_id="anima_layer_lab_gallery",
                        columns=4,
                        preview=True,
                        height=shared.opts.gallery_height or 680,
                        interactive=False,
                        type="pil",
                        object_fit="contain",
                    )
                generation_info = gr.Textbox(
                    label="Generation details",
                    lines=8,
                    max_lines=12,
                    interactive=False,
                    elem_id="anima_layer_lab_generation_info",
                )
                preview_status = gr.HTML("")

        refresh_button.click(
            fn=refresh_choices_ui,
            inputs=[old_model, new_model, modules],
            outputs=[old_model, new_model, modules],
            queue=False,
        )
        inspect_button.click(
            fn=inspect_models_ui,
            inputs=[old_model, new_model],
            outputs=[compatibility],
        )

        preset_outputs = [
            mode,
            non_block_weight,
            llm_shared_weight,
            *layer_sliders,
            *adapter_sliders,
        ]
        preset_layer_graft.click(
            fn=lambda: _preset_values("layer_graft"),
            outputs=preset_outputs,
            queue=False,
        )
        preset_delta.click(fn=lambda: _preset_values("delta"), outputs=preset_outputs, queue=False)
        preset_all_a.click(fn=lambda: _preset_values("all_a"), outputs=preset_outputs, queue=False)
        preset_all_b.click(fn=lambda: _preset_values("all_b"), outputs=preset_outputs, queue=False)
        preset_alternating.click(
            fn=lambda: _preset_values("alternating"),
            outputs=preset_outputs,
            queue=False,
        )
        preset_early_late.click(
            fn=lambda: _preset_values("early_a_late_b"),
            outputs=preset_outputs,
            queue=False,
        )
        apply_groups.click(
            fn=_apply_group_weights,
            inputs=[original_group, inserted_group],
            outputs=layer_sliders,
            queue=False,
        )

        config_outputs = [
            old_model,
            new_model,
            output_name,
            mode,
            output_dtype,
            overwrite,
            non_block_weight,
            llm_shared_weight,
            *layer_sliders,
            *adapter_sliders,
            config_status,
        ]
        load_config_button.click(
            fn=load_merge_config_ui,
            inputs=[merge_config_upload, merge_config_path],
            outputs=config_outputs,
        )

        build_inputs = [
            old_model,
            new_model,
            output_name,
            mode,
            output_dtype,
            overwrite,
            non_block_weight,
            llm_shared_weight,
            *layer_sliders,
            *adapter_sliders,
        ]
        build_button.click(
            fn=call_queue.wrap_gradio_gpu_call(
                build_merge_ui,
                extra_outputs=[gr.skip(), gr.skip(), gr.skip()],
            ),
            inputs=build_inputs,
            outputs=[
                built_checkpoint,
                built_display,
                generated_config,
                build_status,
            ],
        )

        preview_inputs = [
            modules,
            prompt,
            negative_prompt,
            seed,
            width,
            height,
            steps,
            cfg,
            shift,
            sampler,
            scheduler,
            save_preview,
            restore_loaded_model,
        ]
        preview_button.click(
            fn=call_queue.wrap_gradio_gpu_call(preview_merged_ui, extra_outputs=[[], gr.skip()]),
            inputs=[built_checkpoint, *preview_inputs],
            outputs=[gallery, generation_info, preview_status],
        )
        compare_button.click(
            fn=call_queue.wrap_gradio_gpu_call(compare_ui, extra_outputs=[[], gr.skip()]),
            inputs=[old_model, new_model, built_checkpoint, *preview_inputs],
            outputs=[gallery, generation_info, preview_status],
        )

        for component in (
            old_model,
            new_model,
            modules,
            mode,
            output_name,
            output_dtype,
            overwrite,
            non_block_weight,
            llm_shared_weight,
            *layer_sliders,
            *adapter_sliders,
        ):
            component.do_not_save_to_config = True

    return [(interface, "Anima Layer Lab", "anima_layer_lab_tab")]

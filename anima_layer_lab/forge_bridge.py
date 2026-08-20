from __future__ import annotations

import os
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ForgeChoices:
    checkpoints: tuple[str, ...]
    modules: tuple[str, ...]
    default_modules: tuple[str, ...]


def refresh_forge_choices() -> ForgeChoices:
    from modules_forge import main_entry

    checkpoints, modules = main_entry.refresh_models()
    defaults: list[str] = []
    for module in modules:
        lowered = os.path.basename(module).lower()
        if "qwen_3_06b" in lowered or "qwen3_06b" in lowered:
            defaults.append(module)
        elif "qwen_image_vae" in lowered:
            defaults.append(module)
    return ForgeChoices(
        checkpoints=tuple(checkpoints),
        modules=tuple(modules),
        default_modules=tuple(defaults),
    )


def resolve_checkpoint(value: str):
    from modules import sd_models

    if not value:
        raise ValueError("Select a checkpoint")
    match = sd_models.get_closet_checkpoint_match(value)
    if match is not None:
        return match
    wanted = os.path.normcase(os.path.abspath(value))
    for info in sd_models.checkpoints_list.values():
        if os.path.normcase(os.path.abspath(info.filename)) == wanted:
            return info
    raise ValueError(f"Forge checkpoint not found: {value}")


def checkpoint_path(value: str) -> str:
    return str(Path(resolve_checkpoint(value).filename).resolve())


def register_checkpoint(path: str) -> str:
    from modules import sd_models

    sd_models.list_models()
    info = resolve_checkpoint(path)
    return info.name


def model_output_path(filename: str) -> str:
    from modules import sd_models

    clean = os.path.basename((filename or "anima-layer-merge.safetensors").strip())
    if not clean.lower().endswith(".safetensors"):
        clean += ".safetensors"
    if clean in (".safetensors", "..safetensors"):
        raise ValueError("Enter a valid output filename")
    return os.path.join(sd_models.model_path, "AnimaLayerLab", clean)


def sampler_choices() -> tuple[list[str], list[str]]:
    from modules import sd_samplers, sd_schedulers

    samplers = [item.name for item in sd_samplers.visible_samplers()]
    schedulers = [item.label for item in sd_schedulers.schedulers]
    return samplers, schedulers


def _generate_one(
    checkpoint: str,
    modules: list[str],
    prompt: str,
    negative_prompt: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    shift: float,
    sampler: str,
    scheduler: str,
    save_preview: bool,
    username: str | None,
):
    from modules import processing, shared

    checkpoint_info = resolve_checkpoint(checkpoint)
    override_settings = {
        "sd_model_checkpoint": checkpoint_info.name,
        "forge_additional_modules": list(modules),
    }
    p = processing.StableDiffusionProcessingTxt2Img(
        outpath_samples=shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples,
        outpath_grids=shared.opts.outdir_grids or shared.opts.outdir_txt2img_grids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=int(seed),
        batch_size=1,
        n_iter=1,
        steps=int(steps),
        cfg_scale=float(cfg),
        distilled_cfg_scale=float(shift),
        width=int(width),
        height=int(height),
        sampler_name=sampler,
        scheduler=scheduler,
        do_not_save_samples=not save_preview,
        do_not_save_grid=True,
        override_settings=override_settings,
    )
    p.override_settings_restore_afterwards = True
    p.user = username
    with closing(p):
        result = processing.process_images(p)
    if not result.images:
        raise RuntimeError(f"Forge returned no image for {checkpoint_info.name}")
    infotext = result.infotexts[0] if result.infotexts else result.info
    return result.images[0], infotext, checkpoint_info.name


def _generate_preview_impl(
    checkpoints: Iterable[tuple[str, str]],
    modules: list[str],
    prompt: str,
    negative_prompt: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    shift: float,
    sampler: str,
    scheduler: str,
    save_preview: bool,
    restore_loaded_model: bool,
    username: str | None,
):
    import secrets

    from modules import sd_models, shared

    gallery = []
    details: list[str] = []
    if int(seed) < 0:
        # Resolve one random seed for the whole comparison, otherwise each model
        # would receive a different random seed and the comparison would be noisy.
        seed = secrets.randbelow(2**32)
    try:
        for label, checkpoint in checkpoints:
            image, infotext, resolved_name = _generate_one(
                checkpoint=checkpoint,
                modules=modules,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                width=width,
                height=height,
                steps=steps,
                cfg=cfg,
                shift=shift,
                sampler=sampler,
                scheduler=scheduler,
                save_preview=save_preview,
                username=username,
            )
            gallery.append((image, f"{label} — {resolved_name}"))
            details.append(f"{label}\n{infotext}")
    finally:
        shared.total_tqdm.clear()
        if restore_loaded_model:
            # process_images already restored the saved options and loading parameters;
            # this optional reload also restores the physical model in RAM/VRAM now.
            sd_models.forge_model_reload()
    return gallery, "\n\n".join(details), ""


def generate_preview(
    checkpoints: Iterable[tuple[str, str]],
    modules: list[str],
    prompt: str,
    negative_prompt: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    shift: float,
    sampler: str,
    scheduler: str,
    save_preview: bool,
    restore_loaded_model: bool,
    username: str | None = None,
):
    from modules_forge import main_thread

    return main_thread.run_and_wait_result(
        _generate_preview_impl,
        tuple(checkpoints),
        list(modules or []),
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
        username,
    )

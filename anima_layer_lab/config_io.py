from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from . import __version__
from .constants import MERGE_MODES, NEW_BLOCK_COUNT, OUTPUT_DTYPES
from .inspection import read_safetensors_header
from .merge import MergeRecipe


CONFIG_FORMAT = "anima-layer-lab-merge-config"
CONFIG_VERSION = 1
METADATA_KEY = "anima_layer_lab_recipe"


def config_path_for_checkpoint(checkpoint_path: str | os.PathLike[str]) -> str:
    path = Path(checkpoint_path).expanduser().resolve()
    if path.suffix.lower() == ".safetensors":
        stem = path.with_suffix("")
    else:
        stem = path
    return str(stem.parent / f"{stem.name}_merge_config.json")


def config_document(recipe: MergeRecipe) -> dict[str, Any]:
    return {
        "format": CONFIG_FORMAT,
        "format_version": CONFIG_VERSION,
        "extension_version": __version__,
        "recipe": recipe.metadata_dict(),
    }


def write_merge_config(recipe: MergeRecipe) -> str:
    path = config_path_for_checkpoint(recipe.output_path)
    part_path = path + ".part"
    document = config_document(recipe)
    try:
        with open(part_path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(part_path, path)
    except Exception:
        if os.path.exists(part_path):
            os.remove(part_path)
        raise
    return path


def _recipe_from_document(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ValueError("Merge config must contain a JSON object")
    if "recipe" in document:
        if document.get("format") not in (None, CONFIG_FORMAT):
            raise ValueError(
                f"Unsupported merge config format: {document.get('format')}"
            )
        document = document["recipe"]
    if not isinstance(document, dict):
        raise ValueError("Merge config recipe must be a JSON object")
    return document


def _validate_recipe_data(recipe: dict[str, Any]) -> dict[str, Any]:
    required = {
        "old_path",
        "new_path",
        "output_path",
        "mode",
        "layer_weights",
        "adapter_weights",
        "non_block_weight",
        "llm_shared_weight",
        "output_dtype",
    }
    missing = sorted(required - recipe.keys())
    if missing:
        raise ValueError(f"Merge config is missing: {', '.join(missing)}")
    if recipe["mode"] not in MERGE_MODES:
        raise ValueError(f"Unknown merge mode: {recipe['mode']}")
    if recipe["output_dtype"] not in OUTPUT_DTYPES:
        raise ValueError(f"Unknown output dtype: {recipe['output_dtype']}")

    layer_weights = tuple(float(value) for value in recipe["layer_weights"])
    adapter_weights = tuple(float(value) for value in recipe["adapter_weights"])
    if len(layer_weights) != NEW_BLOCK_COUNT:
        raise ValueError(f"Merge config must contain {NEW_BLOCK_COUNT} layer weights")
    if len(adapter_weights) != 6:
        raise ValueError("Merge config must contain six LLM-adapter weights")

    normalized = dict(recipe)
    normalized["layer_weights"] = layer_weights
    normalized["adapter_weights"] = adapter_weights
    normalized["non_block_weight"] = float(recipe["non_block_weight"])
    normalized["llm_shared_weight"] = float(recipe["llm_shared_weight"])
    validation_recipe = MergeRecipe(
        old_path=str(normalized["old_path"]),
        new_path=str(normalized["new_path"]),
        output_path=str(normalized["output_path"]),
        mode=str(normalized["mode"]),
        layer_weights=layer_weights,
        adapter_weights=adapter_weights,
        non_block_weight=normalized["non_block_weight"],
        llm_shared_weight=normalized["llm_shared_weight"],
        output_dtype=str(normalized["output_dtype"]),
    )
    validation_recipe.validate()
    return normalized


def read_merge_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    resolved = str(Path(path).expanduser().resolve())
    if resolved.lower().endswith(".safetensors"):
        _, metadata, _ = read_safetensors_header(resolved)
        raw_recipe = metadata.get(METADATA_KEY)
        if not raw_recipe:
            raise ValueError("Safetensors file has no Anima Layer Lab merge recipe")
        try:
            document = json.loads(raw_recipe)
        except json.JSONDecodeError as exc:
            raise ValueError("Safetensors merge recipe contains invalid JSON") from exc
    elif resolved.lower().endswith(".json"):
        with open(resolved, encoding="utf-8") as handle:
            document = json.load(handle)
    else:
        raise ValueError("Select a .json merge config or merged .safetensors file")
    return _validate_recipe_data(_recipe_from_document(document))

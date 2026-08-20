from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from . import __version__
from .constants import (
    LAYER_ORIGINS,
    MERGE_MODE_INTERWEAVE,
    MERGE_MODE_WEIGHTED,
    MERGE_MODES,
    NEW_BLOCK_COUNT,
    OLD_TO_NEW,
    OUTPUT_DTYPE_BF16,
    OUTPUT_DTYPE_FP16,
    OUTPUT_DTYPE_MATCH_B,
    OUTPUT_DTYPES,
    is_muted_output_suffix,
)
from .inspection import (
    DTYPE_BYTES,
    SUPPORTED_FLOAT_DTYPES,
    CheckpointSummary,
    TensorInfo,
    normalize_key,
    parse_block_key,
    validate_pair,
)

ProgressCallback = Callable[[int, int, str], bool | None]


class MergeCancelled(RuntimeError):
    pass


@dataclass
class MergeRecipe:
    old_path: str
    new_path: str
    output_path: str
    mode: str = MERGE_MODE_WEIGHTED
    layer_weights: tuple[float, ...] = tuple(
        1.0 if origin.inserted else 0.0 for origin in LAYER_ORIGINS
    )
    adapter_weights: tuple[float, ...] = (0.0,) * 6
    non_block_weight: float = 0.0
    llm_shared_weight: float = 0.0
    output_dtype: str = OUTPUT_DTYPE_MATCH_B
    overwrite: bool = False

    def validate(self) -> None:
        if self.mode not in MERGE_MODES:
            raise ValueError(f"Unknown merge mode: {self.mode}")
        if self.output_dtype not in OUTPUT_DTYPES:
            raise ValueError(f"Unknown output dtype: {self.output_dtype}")
        if len(self.layer_weights) != NEW_BLOCK_COUNT:
            raise ValueError(f"Expected {NEW_BLOCK_COUNT} layer weights")
        if len(self.adapter_weights) != 6:
            raise ValueError("Expected six LLM-adapter weights")
        for label, values in (
            ("layer", self.layer_weights),
            ("adapter", self.adapter_weights),
        ):
            for index, value in enumerate(values):
                if not 0.0 <= float(value) <= 1.0:
                    raise ValueError(f"{label} weight {index} is outside 0..1")
        for label, value in (
            ("non-block", self.non_block_weight),
            ("LLM shared", self.llm_shared_weight),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{label} weight is outside 0..1")

    def metadata_dict(self) -> dict:
        data = asdict(self)
        data.pop("overwrite", None)
        data["old_path"] = os.path.basename(self.old_path)
        data["new_path"] = os.path.basename(self.new_path)
        data["output_path"] = os.path.basename(self.output_path)
        data["layer_weights"] = [round(float(v), 6) for v in self.layer_weights]
        data["adapter_weights"] = [round(float(v), 6) for v in self.adapter_weights]
        return data


@dataclass(frozen=True)
class BuildResult:
    output_path: str
    file_size: int
    tensor_count: int
    recipe: MergeRecipe


@dataclass(frozen=True)
class OutputTensorSpec:
    key: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]

    @property
    def nbytes(self) -> int:
        return self.data_offsets[1] - self.data_offsets[0]


_SAFE_TO_TORCH = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I32": torch.int32,
    "F32": torch.float32,
    "I64": torch.int64,
    "F64": torch.float64,
}


def _clamp_weight(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _output_dtype(source_dtype: str, selection: str) -> str:
    if source_dtype not in SUPPORTED_FLOAT_DTYPES:
        return source_dtype
    if selection == OUTPUT_DTYPE_BF16:
        return "BF16"
    if selection == OUTPUT_DTYPE_FP16:
        return "F16"
    return source_dtype


def _tensor_nbytes(shape: tuple[int, ...], dtype: str) -> int:
    count = 1
    for dimension in shape:
        count *= dimension
    return count * DTYPE_BYTES[dtype]


def _sanitize_metadata(metadata: dict) -> dict[str, str]:
    sanitized: dict[str, str] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, str):
            sanitized[str(key)] = value
        elif isinstance(value, (dict, list, tuple)):
            sanitized[str(key)] = json.dumps(value, separators=(",", ":"))
        else:
            sanitized[str(key)] = str(value)
    return sanitized


def _build_output_header(
    new: CheckpointSummary,
    recipe: MergeRecipe,
) -> tuple[bytes, list[OutputTensorSpec], int]:
    specs: list[OutputTensorSpec] = []
    offset = 0
    for key, info in new.tensors.items():
        dtype = _output_dtype(info.dtype, recipe.output_dtype)
        nbytes = _tensor_nbytes(info.shape, dtype)
        specs.append(
            OutputTensorSpec(
                key=key,
                dtype=dtype,
                shape=info.shape,
                data_offsets=(offset, offset + nbytes),
            ),
        )
        offset += nbytes

    metadata = dict(new.metadata)
    metadata.update(
        {
            "anima_layer_lab_version": __version__,
            "anima_layer_lab_recipe": json.dumps(recipe.metadata_dict(), separators=(",", ":")),
            "anima_layer_lab_old_model": os.path.basename(recipe.old_path),
            "anima_layer_lab_new_model": os.path.basename(recipe.new_path),
        },
    )
    header: dict = {"__metadata__": _sanitize_metadata(metadata)}
    for spec in specs:
        header[spec.key] = {
            "dtype": spec.dtype,
            "shape": list(spec.shape),
            "data_offsets": list(spec.data_offsets),
        }

    raw = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    padding = (-len(raw)) % 8
    raw += b" " * padding
    return raw, specs, offset


def _copy_raw_tensor(
    source_handle,
    source_summary: CheckpointSummary,
    source_info: TensorInfo,
    output_handle,
    output_absolute_offset: int,
    chunk_bytes: int = 16 * 1024 * 1024,
) -> None:
    source_handle.seek(source_summary.data_start + source_info.data_offsets[0])
    output_handle.seek(output_absolute_offset)
    remaining = source_info.nbytes
    while remaining:
        data = source_handle.read(min(chunk_bytes, remaining))
        if not data:
            raise OSError(
                f"Unexpected end of {source_summary.path} while copying {source_info.key}",
            )
        output_handle.write(data)
        remaining -= len(data)


def _write_tensor(output_handle, absolute_offset: int, tensor: torch.Tensor, dtype: str) -> None:
    torch_dtype = _SAFE_TO_TORCH.get(dtype)
    if torch_dtype is None:
        raise ValueError(f"Cannot write dtype {dtype}")
    converted = tensor.detach().to(device="cpu", dtype=torch_dtype).contiguous()
    raw_view = converted.view(torch.uint8).numpy()
    output_handle.seek(absolute_offset)
    output_handle.write(raw_view.tobytes())


def _write_zeros(output_handle, absolute_offset: int, nbytes: int) -> None:
    output_handle.seek(absolute_offset)
    chunk = b"\0" * min(nbytes, 16 * 1024 * 1024)
    remaining = nbytes
    while remaining:
        portion = chunk if remaining >= len(chunk) else chunk[:remaining]
        output_handle.write(portion)
        remaining -= len(portion)


def _can_raw_copy(source_info: TensorInfo, output_spec: OutputTensorSpec) -> bool:
    return source_info.dtype == output_spec.dtype and source_info.shape == output_spec.shape


def _blend_tensors(a: torch.Tensor, b: torch.Tensor, weight: float) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch during merge: {tuple(a.shape)} != {tuple(b.shape)}")
    if not (a.is_floating_point() and b.is_floating_point()):
        return b if weight >= 0.5 else a
    weight = _clamp_weight(weight)
    if weight <= 0.0:
        return a
    if weight >= 1.0:
        return b
    return torch.lerp(a.float(), b.float(), weight)


def _adapter_index(canonical: str) -> int | None:
    prefix = "llm_adapter.blocks."
    if not canonical.startswith(prefix):
        return None
    remainder = canonical[len(prefix) :]
    index_text = remainder.split(".", 1)[0]
    return int(index_text) if index_text.isdigit() else None


def _non_block_weight(recipe: MergeRecipe, canonical: str) -> float:
    adapter_index = _adapter_index(canonical)
    if adapter_index is not None and adapter_index < len(recipe.adapter_weights):
        return _clamp_weight(recipe.adapter_weights[adapter_index])
    if canonical.startswith("llm_adapter."):
        return _clamp_weight(recipe.llm_shared_weight)
    return _clamp_weight(recipe.non_block_weight)


def _write_direct_choice(
    *,
    use_new: bool,
    old_key: str,
    new_key: str,
    old: CheckpointSummary,
    new: CheckpointSummary,
    old_safe,
    new_safe,
    old_raw,
    new_raw,
    output_handle,
    output_spec: OutputTensorSpec,
    output_absolute_offset: int,
    old_is_zero: bool = False,
) -> None:
    if not use_new and old_is_zero:
        _write_zeros(output_handle, output_absolute_offset, output_spec.nbytes)
        return
    summary = new if use_new else old
    key = new_key if use_new else old_key
    safe = new_safe if use_new else old_safe
    raw = new_raw if use_new else old_raw
    info = summary.tensors[key]
    if _can_raw_copy(info, output_spec):
        _copy_raw_tensor(raw, summary, info, output_handle, output_absolute_offset)
    else:
        _write_tensor(
            output_handle,
            output_absolute_offset,
            safe.get_tensor(key),
            output_spec.dtype,
        )


def build_merge(
    recipe: MergeRecipe,
    progress: ProgressCallback | None = None,
) -> BuildResult:
    recipe.validate()
    report = validate_pair(recipe.old_path, recipe.new_path)
    if not report.valid or report.old is None or report.new is None:
        raise ValueError("Checkpoint pair is incompatible:\n" + "\n".join(report.errors))
    old = report.old
    new = report.new

    output = str(Path(recipe.output_path).expanduser().resolve())
    if not output.lower().endswith(".safetensors"):
        output += ".safetensors"
    output_dir = os.path.dirname(output)
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(output) and not recipe.overwrite:
        raise FileExistsError(f"Output already exists: {output}")

    raw_header, specs, data_bytes = _build_output_header(new, recipe)
    total_size = 8 + len(raw_header) + data_bytes
    free_bytes = shutil.disk_usage(output_dir).free
    if free_bytes < int(total_size * 1.05):
        raise OSError(
            f"Not enough free disk space: need about {total_size / 2**30:.2f} GiB, "
            f"have {free_bytes / 2**30:.2f} GiB",
        )

    # Keep the safetensors suffix so the completed temporary file can be parsed
    # by the same strict validation path before it is atomically published.
    part_path = output + ".part.safetensors"
    if os.path.exists(part_path):
        os.remove(part_path)

    output_data_start = 8 + len(raw_header)
    try:
        with (
            safe_open(old.path, framework="pt", device="cpu") as old_safe,
            safe_open(new.path, framework="pt", device="cpu") as new_safe,
            open(old.path, "rb") as old_raw,
            open(new.path, "rb") as new_raw,
            open(part_path, "w+b") as output_handle,
            torch.inference_mode(),
        ):
            output_handle.write(len(raw_header).to_bytes(8, "little"))
            output_handle.write(raw_header)
            output_handle.truncate(total_size)

            total = len(specs)
            for position, spec in enumerate(specs, start=1):
                if progress is not None and progress(position - 1, total, spec.key) is False:
                    raise MergeCancelled("Merge cancelled")

                new_info = new.tensors[spec.key]
                parsed_block = parse_block_key(spec.key, new.block_prefix)
                absolute_offset = output_data_start + spec.data_offsets[0]

                if parsed_block is not None:
                    block_index, suffix = parsed_block
                    origin = LAYER_ORIGINS[block_index]
                    source_old = (
                        origin.old_index
                        if origin.old_index is not None
                        else origin.source_old_index
                    )
                    assert source_old is not None
                    old_key = old.block_key(source_old, suffix)
                    if old_key is None:
                        raise KeyError(f"Missing old tensor blocks.{source_old}{suffix}")
                    weight = _clamp_weight(recipe.layer_weights[block_index])
                    old_zero = origin.inserted and is_muted_output_suffix(suffix)

                    if recipe.mode == MERGE_MODE_INTERWEAVE:
                        _write_direct_choice(
                            use_new=weight >= 0.5,
                            old_key=old_key,
                            new_key=spec.key,
                            old=old,
                            new=new,
                            old_safe=old_safe,
                            new_safe=new_safe,
                            old_raw=old_raw,
                            new_raw=new_raw,
                            output_handle=output_handle,
                            output_spec=spec,
                            output_absolute_offset=absolute_offset,
                            old_is_zero=old_zero,
                        )
                        continue

                    if recipe.mode == MERGE_MODE_WEIGHTED or not origin.inserted:
                        if weight <= 0.0 or weight >= 1.0:
                            _write_direct_choice(
                                use_new=weight >= 1.0,
                                old_key=old_key,
                                new_key=spec.key,
                                old=old,
                                new=new,
                                old_safe=old_safe,
                                new_safe=new_safe,
                                old_raw=old_raw,
                                new_raw=new_raw,
                                output_handle=output_handle,
                                output_spec=spec,
                                output_absolute_offset=absolute_offset,
                                old_is_zero=old_zero,
                            )
                        else:
                            old_tensor = (
                                torch.zeros(new_info.shape, dtype=torch.float32)
                                if old_zero
                                else old_safe.get_tensor(old_key)
                            )
                            merged = _blend_tensors(
                                old_tensor, new_safe.get_tensor(spec.key), weight,
                            )
                            _write_tensor(output_handle, absolute_offset, merged, spec.dtype)
                        continue

                    # Experimental relative delta graft for an inserted block.
                    # The final release does not contain the pre-training snapshot,
                    # so B's current source block is the only available baseline:
                    # A_source + w * (B_inserted - B_source).
                    source_new = OLD_TO_NEW[source_old]
                    new_source_key = new.block_key(source_new, suffix)
                    if new_source_key is None:
                        raise KeyError(
                            f"Missing new initialization tensor blocks.{source_new}{suffix}",
                        )
                    trained = new_safe.get_tensor(spec.key)
                    if old_zero:
                        merged = trained.float().mul(weight)
                    else:
                        old_init = old_safe.get_tensor(old_key).float()
                        new_init = new_safe.get_tensor(new_source_key).float()
                        merged = old_init.add((trained.float() - new_init).mul(weight))
                    _write_tensor(output_handle, absolute_offset, merged, spec.dtype)
                    continue

                canonical = normalize_key(spec.key)
                old_key = old.canonical_to_actual.get(canonical)
                if old_key is None:
                    if _can_raw_copy(new_info, spec):
                        _copy_raw_tensor(new_raw, new, new_info, output_handle, absolute_offset)
                    else:
                        _write_tensor(
                            output_handle,
                            absolute_offset,
                            new_safe.get_tensor(spec.key),
                            spec.dtype,
                        )
                    continue

                weight = _non_block_weight(recipe, canonical)
                if recipe.mode == MERGE_MODE_INTERWEAVE or weight <= 0.0 or weight >= 1.0:
                    _write_direct_choice(
                        use_new=(
                            weight >= 0.5 if recipe.mode == MERGE_MODE_INTERWEAVE else weight >= 1.0
                        ),
                        old_key=old_key,
                        new_key=spec.key,
                        old=old,
                        new=new,
                        old_safe=old_safe,
                        new_safe=new_safe,
                        old_raw=old_raw,
                        new_raw=new_raw,
                        output_handle=output_handle,
                        output_spec=spec,
                        output_absolute_offset=absolute_offset,
                    )
                else:
                    merged = _blend_tensors(
                        old_safe.get_tensor(old_key),
                        new_safe.get_tensor(spec.key),
                        weight,
                    )
                    _write_tensor(output_handle, absolute_offset, merged, spec.dtype)

            output_handle.flush()
            os.fsync(output_handle.fileno())

        # Re-read the header before exposing the result to Forge.
        validation = validate_pair(recipe.old_path, part_path)
        if validation.new is None or validation.new.block_count != NEW_BLOCK_COUNT:
            raise OSError("Completed output failed safetensors validation")
        if os.path.exists(output) and not recipe.overwrite:
            raise FileExistsError(f"Output appeared while merging: {output}")
        os.replace(part_path, output)
    except Exception:
        if os.path.exists(part_path):
            os.remove(part_path)
        raise

    if progress is not None:
        progress(len(specs), len(specs), "Complete")
    return BuildResult(
        output_path=output,
        file_size=os.path.getsize(output),
        tensor_count=len(specs),
        recipe=recipe,
    )

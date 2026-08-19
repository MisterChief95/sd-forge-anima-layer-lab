from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from math import prod
from pathlib import Path

from .constants import LAYER_ORIGINS, NEW_BLOCK_COUNT, OLD_BLOCK_COUNT, OLD_TO_NEW


MAX_HEADER_BYTES = 128 * 1024 * 1024
_BLOCK_RE = re.compile(r"^(.*?blocks\.)(\d+)(\..+)$")
_ADAPTER_RE = re.compile(r"^llm_adapter\.blocks\.(\d+)(\..+)$")
_ROOT_PREFIXES = (
    "model.diffusion_model.",
    "diffusion_model.",
    "net.",
)

DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
SUPPORTED_FLOAT_DTYPES = frozenset({"F16", "BF16", "F32", "F64"})


class CheckpointFormatError(ValueError):
    pass


@dataclass(frozen=True)
class TensorInfo:
    key: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]

    @property
    def nbytes(self) -> int:
        return self.data_offsets[1] - self.data_offsets[0]

    @property
    def parameter_count(self) -> int:
        return prod(self.shape) if self.shape else 1


@dataclass
class CheckpointSummary:
    path: str
    file_size: int
    header_size: int
    metadata: dict[str, str]
    tensors: dict[str, TensorInfo]
    block_prefix: str
    block_count: int
    block_suffixes: dict[int, frozenset[str]]
    canonical_to_actual: dict[str, str]
    adapter_block_count: int
    parameter_count: int
    dtypes: tuple[str, ...]
    warnings: list[str] = field(default_factory=list)

    @property
    def data_start(self) -> int:
        return 8 + self.header_size

    @property
    def filename(self) -> str:
        return os.path.basename(self.path)

    def block_key(self, index: int, suffix: str) -> str | None:
        return self.canonical_to_actual.get(f"blocks.{index}{suffix}")

    def canonical_key(self, key: str) -> str:
        parsed = parse_block_key(key, self.block_prefix)
        if parsed is not None:
            index, suffix = parsed
            return f"blocks.{index}{suffix}"
        return normalize_key(key)


@dataclass
class CompatibilityReport:
    valid: bool
    errors: list[str]
    warnings: list[str]
    matched_tensor_count: int
    checked_block_tensor_count: int
    old: CheckpointSummary | None = None
    new: CheckpointSummary | None = None


def normalize_key(key: str) -> str:
    normalized = key
    changed = True
    while changed:
        changed = False
        for prefix in _ROOT_PREFIXES:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                changed = True
                break
    return normalized


def read_safetensors_header(
    path: str | os.PathLike[str],
) -> tuple[int, dict, dict[str, TensorInfo]]:
    resolved = str(Path(path).expanduser().resolve())
    if not os.path.isfile(resolved):
        raise FileNotFoundError(resolved)
    if not resolved.lower().endswith(".safetensors"):
        raise CheckpointFormatError("Only .safetensors checkpoints are supported")

    file_size = os.path.getsize(resolved)
    with open(resolved, "rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise CheckpointFormatError("File is too small to be safetensors")
        header_size = int.from_bytes(raw_length, "little", signed=False)
        if header_size <= 2 or header_size > MAX_HEADER_BYTES:
            raise CheckpointFormatError(
                f"Invalid safetensors header size: {header_size}"
            )
        raw_header = handle.read(header_size)
        if len(raw_header) != header_size:
            raise CheckpointFormatError("Truncated safetensors header")

    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointFormatError(f"Invalid safetensors JSON header: {exc}") from exc

    metadata = header.pop("__metadata__", {}) or {}
    if not isinstance(metadata, dict):
        raise CheckpointFormatError("Safetensors metadata must be an object")

    tensors: dict[str, TensorInfo] = {}
    data_size = file_size - 8 - header_size
    for key, value in header.items():
        if not isinstance(value, dict):
            raise CheckpointFormatError(f"Invalid tensor entry for {key}")
        try:
            dtype = str(value["dtype"])
            shape = tuple(int(v) for v in value["shape"])
            offsets = tuple(int(v) for v in value["data_offsets"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointFormatError(f"Invalid tensor descriptor for {key}") from exc
        if (
            len(offsets) != 2
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or offsets[1] > data_size
        ):
            raise CheckpointFormatError(f"Invalid data offsets for {key}: {offsets}")
        if dtype not in DTYPE_BYTES:
            raise CheckpointFormatError(
                f"Unsupported safetensors dtype {dtype} in {key}"
            )
        expected = (prod(shape) if shape else 1) * DTYPE_BYTES[dtype]
        if expected != offsets[1] - offsets[0]:
            raise CheckpointFormatError(
                f"Tensor byte count mismatch for {key}: expected {expected}, found {offsets[1] - offsets[0]}"
            )
        tensors[key] = TensorInfo(
            key=key, dtype=dtype, shape=shape, data_offsets=offsets
        )

    if not tensors:
        raise CheckpointFormatError("Checkpoint contains no tensors")
    return header_size, {str(k): str(v) for k, v in metadata.items()}, tensors


def detect_block_prefix(
    keys: list[str] | tuple[str, ...],
) -> tuple[str, int, dict[int, frozenset[str]]]:
    groups: dict[str, dict[int, set[str]]] = {}
    for key in keys:
        match = _BLOCK_RE.match(key)
        if match is None:
            continue
        prefix, raw_index, suffix = match.groups()
        if "llm_adapter." in prefix:
            continue
        groups.setdefault(prefix, {}).setdefault(int(raw_index), set()).add(suffix)

    candidates: list[tuple[tuple[int, int, int], str, dict[int, set[str]]]] = []
    for prefix, by_index in groups.items():
        indexes = sorted(by_index)
        if not indexes or indexes != list(range(indexes[-1] + 1)):
            continue
        suffixes = set().union(*by_index.values())
        marker = int(".mlp.layer1.weight" in suffixes)
        expected = int(len(indexes) in (OLD_BLOCK_COUNT, NEW_BLOCK_COUNT))
        score = (expected, marker, sum(len(v) for v in by_index.values()))
        candidates.append((score, prefix, by_index))

    if not candidates:
        raise CheckpointFormatError(
            "Could not find a contiguous Anima transformer block stack"
        )
    candidates.sort(reverse=True, key=lambda item: item[0])
    _, prefix, by_index = candidates[0]
    return (
        prefix,
        len(by_index),
        {index: frozenset(values) for index, values in by_index.items()},
    )


def parse_block_key(key: str, block_prefix: str) -> tuple[int, str] | None:
    if not key.startswith(block_prefix):
        return None
    remainder = key[len(block_prefix) :]
    index_text, separator, suffix = remainder.partition(".")
    if not separator or not index_text.isdigit():
        return None
    return int(index_text), f".{suffix}"


def inspect_checkpoint(path: str | os.PathLike[str]) -> CheckpointSummary:
    resolved = str(Path(path).expanduser().resolve())
    header_size, metadata, tensors = read_safetensors_header(resolved)
    block_prefix, block_count, block_suffixes = detect_block_prefix(
        tuple(tensors.keys())
    )

    canonical_to_actual: dict[str, str] = {}
    warnings: list[str] = []
    adapter_indexes: set[int] = set()
    for key in tensors:
        parsed = parse_block_key(key, block_prefix)
        if parsed is not None:
            index, suffix = parsed
            canonical = f"blocks.{index}{suffix}"
        else:
            canonical = normalize_key(key)
            adapter_match = _ADAPTER_RE.match(canonical)
            if adapter_match:
                adapter_indexes.add(int(adapter_match.group(1)))
        if canonical in canonical_to_actual:
            warnings.append(f"Duplicate canonical tensor key: {canonical}")
        canonical_to_actual[canonical] = key

    return CheckpointSummary(
        path=resolved,
        file_size=os.path.getsize(resolved),
        header_size=header_size,
        metadata=metadata,
        tensors=tensors,
        block_prefix=block_prefix,
        block_count=block_count,
        block_suffixes=block_suffixes,
        canonical_to_actual=canonical_to_actual,
        adapter_block_count=(max(adapter_indexes) + 1 if adapter_indexes else 0),
        parameter_count=sum(info.parameter_count for info in tensors.values()),
        dtypes=tuple(sorted({info.dtype for info in tensors.values()})),
        warnings=warnings,
    )


def _quantization_errors(summary: CheckpointSummary, label: str) -> list[str]:
    errors: list[str] = []
    for key, info in summary.tensors.items():
        if info.dtype not in SUPPORTED_FLOAT_DTYPES:
            errors.append(
                f"{label} uses unsupported/quantized dtype {info.dtype} in {key}"
            )
            break
        lowered = key.lower()
        if any(marker in lowered for marker in ("qweight", "w_scale", "weight_scale")):
            errors.append(
                f"{label} appears quantized ({key}); use a full-precision FP16/BF16 checkpoint"
            )
            break
    return errors


def validate_pair(
    old_path: str | os.PathLike[str], new_path: str | os.PathLike[str]
) -> CompatibilityReport:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        old = inspect_checkpoint(old_path)
    except Exception as exc:
        return CompatibilityReport(False, [f"Old model: {exc}"], [], 0, 0)
    try:
        new = inspect_checkpoint(new_path)
    except Exception as exc:
        return CompatibilityReport(
            False, [f"New model: {exc}"], old.warnings, 0, 0, old=old
        )

    warnings.extend(old.warnings)
    warnings.extend(new.warnings)
    if old.block_count != OLD_BLOCK_COUNT:
        errors.append(
            f"Old model must contain {OLD_BLOCK_COUNT} transformer blocks; found {old.block_count}"
        )
    if new.block_count != NEW_BLOCK_COUNT:
        errors.append(
            f"New model must contain {NEW_BLOCK_COUNT} transformer blocks; found {new.block_count}"
        )
    errors.extend(_quantization_errors(old, "Old model"))
    errors.extend(_quantization_errors(new, "New model"))

    matched = 0
    checked_blocks = 0
    if not errors:
        for origin in LAYER_ORIGINS:
            suffixes = new.block_suffixes.get(origin.new_index, frozenset())
            source_old = (
                origin.old_index
                if origin.old_index is not None
                else origin.source_old_index
            )
            assert source_old is not None
            source_new = OLD_TO_NEW[source_old]
            for suffix in suffixes:
                checked_blocks += 1
                old_key = old.block_key(source_old, suffix)
                if old_key is None:
                    errors.append(f"Old model is missing blocks.{source_old}{suffix}")
                    continue
                new_key = new.block_key(origin.new_index, suffix)
                assert new_key is not None
                if old.tensors[old_key].shape != new.tensors[new_key].shape:
                    errors.append(
                        f"Shape mismatch for output block {origin.new_index}{suffix}: "
                        f"old {old.tensors[old_key].shape}, new {new.tensors[new_key].shape}"
                    )
                    continue
                if origin.inserted:
                    source_key = new.block_key(source_new, suffix)
                    if source_key is None:
                        errors.append(
                            f"New model is missing initialization source blocks.{source_new}{suffix}"
                        )
                        continue
                    if new.tensors[source_key].shape != new.tensors[new_key].shape:
                        errors.append(
                            f"New-model inserted/source shape mismatch at block {origin.new_index}{suffix}"
                        )
                        continue
                matched += 1

        for canonical, new_key in new.canonical_to_actual.items():
            if canonical.startswith("blocks."):
                continue
            old_key = old.canonical_to_actual.get(canonical)
            if old_key is None:
                warnings.append(
                    f"Only the new model contains {canonical}; it will be copied from the new model"
                )
                continue
            if old.tensors[old_key].shape != new.tensors[new_key].shape:
                errors.append(
                    f"Non-block shape mismatch for {canonical}: "
                    f"old {old.tensors[old_key].shape}, new {new.tensors[new_key].shape}"
                )
            else:
                matched += 1

    if len(errors) > 24:
        hidden = len(errors) - 24
        errors = errors[:24] + [f"…and {hidden} more compatibility errors"]

    return CompatibilityReport(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        matched_tensor_count=matched,
        checked_block_tensor_count=checked_blocks,
        old=old,
        new=new,
    )

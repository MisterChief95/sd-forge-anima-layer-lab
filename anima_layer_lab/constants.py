from __future__ import annotations

from dataclasses import dataclass


OLD_BLOCK_COUNT = 28
NEW_BLOCK_COUNT = 40

# Official Anima-2.9B expansion manifest:
# https://huggingface.co/Gazingstars123/Anima-2.9B/blob/main/expand_manifest.json
INSERTION_POSITIONS = (2, 5, 8, 11, 14, 17, 21, 24, 27, 30, 33, 36)
INSERTED_TO_SOURCE = {
    2: 1,
    5: 3,
    8: 5,
    11: 7,
    14: 9,
    17: 11,
    21: 14,
    24: 16,
    27: 18,
    30: 20,
    33: 22,
    36: 24,
}

MUTED_OUTPUT_SUFFIXES = frozenset(
    {
        ".adaln_modulation_self_attn.2.weight",
        ".adaln_modulation_cross_attn.2.weight",
        ".adaln_modulation_mlp.2.weight",
        ".self_attn.output_proj.weight",
        ".cross_attn.output_proj.weight",
        ".mlp.layer2.weight",
    }
)

MERGE_MODE_DELTA = "Relative delta graft (experimental)"
MERGE_MODE_WEIGHTED = "Aligned weighted merge"
MERGE_MODE_INTERWEAVE = "Hard A/B interweave"
MERGE_MODES = (MERGE_MODE_WEIGHTED, MERGE_MODE_INTERWEAVE, MERGE_MODE_DELTA)

OUTPUT_DTYPE_MATCH_B = "Match Anima 2.9B"
OUTPUT_DTYPE_BF16 = "BF16"
OUTPUT_DTYPE_FP16 = "FP16"
OUTPUT_DTYPES = (OUTPUT_DTYPE_MATCH_B, OUTPUT_DTYPE_BF16, OUTPUT_DTYPE_FP16)


@dataclass(frozen=True)
class LayerOrigin:
    new_index: int
    kind: str
    old_index: int | None = None
    source_old_index: int | None = None

    @property
    def inserted(self) -> bool:
        return self.kind == "inserted"


def build_layer_origins() -> tuple[LayerOrigin, ...]:
    origins: list[LayerOrigin] = []
    old_index = 0
    for new_index in range(NEW_BLOCK_COUNT):
        if new_index in INSERTED_TO_SOURCE:
            origins.append(
                LayerOrigin(
                    new_index=new_index,
                    kind="inserted",
                    source_old_index=INSERTED_TO_SOURCE[new_index],
                )
            )
        else:
            origins.append(
                LayerOrigin(
                    new_index=new_index,
                    kind="original",
                    old_index=old_index,
                )
            )
            old_index += 1

    if old_index != OLD_BLOCK_COUNT:
        raise AssertionError(f"Expansion map consumed {old_index} old blocks")
    return tuple(origins)


LAYER_ORIGINS = build_layer_origins()
OLD_TO_NEW = {
    origin.old_index: origin.new_index
    for origin in LAYER_ORIGINS
    if origin.old_index is not None
}


def is_muted_output_suffix(suffix: str) -> bool:
    return suffix in MUTED_OUTPUT_SUFFIXES

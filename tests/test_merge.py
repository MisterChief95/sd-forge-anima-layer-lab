from __future__ import annotations

import os
import tempfile
import unittest

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from anima_layer_lab.config_io import read_merge_config, write_merge_config
from anima_layer_lab.constants import (
    LAYER_ORIGINS,
    MERGE_MODE_DELTA,
    MERGE_MODE_WEIGHTED,
    MUTED_OUTPUT_SUFFIXES,
)
from anima_layer_lab.inspection import inspect_checkpoint, validate_pair
from anima_layer_lab.merge import MergeCancelled, MergeRecipe, build_merge

BLOCK_SUFFIXES = tuple(sorted(MUTED_OUTPUT_SUFFIXES)) + (
    ".mlp.layer1.weight",
    ".self_attn.q_proj.weight",
)


def _value(index: int, suffix_index: int) -> torch.Tensor:
    return torch.full((2, 3), index * 0.125 + suffix_index * 0.03125, dtype=torch.bfloat16)


def make_old_checkpoint(path: str) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for block in range(28):
        for suffix_index, suffix in enumerate(BLOCK_SUFFIXES):
            tensors[f"net.blocks.{block}{suffix}"] = _value(block + 1, suffix_index + 1)
    for block in range(6):
        tensors[f"net.llm_adapter.blocks.{block}.weight"] = torch.full(
            (2, 2),
            block + 0.5,
            dtype=torch.bfloat16,
        )
    tensors["net.llm_adapter.proj.weight"] = torch.full((2, 2), 2.0, dtype=torch.bfloat16)
    tensors["net.x_embedder.proj.1.weight"] = torch.full((2, 2), 3.0, dtype=torch.bfloat16)
    save_file(tensors, path, metadata={"model": "synthetic-old"})
    return tensors


def make_new_checkpoint(path: str, old: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for origin in LAYER_ORIGINS:
        source_old = origin.old_index if origin.old_index is not None else origin.source_old_index
        assert source_old is not None
        for suffix_index, suffix in enumerate(BLOCK_SUFFIXES):
            source = old[f"net.blocks.{source_old}{suffix}"].clone()
            if origin.inserted:
                if suffix in MUTED_OUTPUT_SUFFIXES:
                    source.zero_()
                delta = torch.full_like(source, (origin.new_index + suffix_index + 1) * 0.0078125)
                source.add_(delta)
            tensors[f"net.blocks.{origin.new_index}{suffix}"] = source
    for block in range(6):
        tensors[f"net.llm_adapter.blocks.{block}.weight"] = old[
            f"net.llm_adapter.blocks.{block}.weight"
        ].clone()
    tensors["net.llm_adapter.proj.weight"] = old["net.llm_adapter.proj.weight"].clone()
    tensors["net.x_embedder.proj.1.weight"] = old["net.x_embedder.proj.1.weight"].clone()
    save_file(tensors, path, metadata={"model": "synthetic-new"})
    return tensors


def load_all(path: str) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old_path = os.path.join(self.temp.name, "old.safetensors")
        self.new_path = os.path.join(self.temp.name, "new.safetensors")
        self.old_tensors = make_old_checkpoint(self.old_path)
        self.new_tensors = make_new_checkpoint(self.new_path, self.old_tensors)

    def tearDown(self):
        self.temp.cleanup()

    def test_inspection_and_compatibility(self):
        old = inspect_checkpoint(self.old_path)
        new = inspect_checkpoint(self.new_path)
        self.assertEqual(old.block_count, 28)
        self.assertEqual(new.block_count, 40)
        report = validate_pair(self.old_path, self.new_path)
        self.assertTrue(report.valid, report.errors)

    def test_all_b_weighted_is_tensor_identical(self):
        output = os.path.join(self.temp.name, "all-b.safetensors")
        build_merge(
            MergeRecipe(
                old_path=self.old_path,
                new_path=self.new_path,
                output_path=output,
                mode=MERGE_MODE_WEIGHTED,
                layer_weights=(1.0,) * 40,
                adapter_weights=(1.0,) * 6,
                non_block_weight=1.0,
                llm_shared_weight=1.0,
            ),
        )
        actual = load_all(output)
        expected = load_all(self.new_path)
        self.assertEqual(actual.keys(), expected.keys())
        for key in actual:
            self.assertTrue(torch.equal(actual[key], expected[key]), key)

    def test_relative_delta_matches_new_when_sources_are_unchanged(self):
        output = os.path.join(self.temp.name, "delta.safetensors")
        layer_weights = tuple(1.0 if origin.inserted else 0.0 for origin in LAYER_ORIGINS)
        build_merge(
            MergeRecipe(
                old_path=self.old_path,
                new_path=self.new_path,
                output_path=output,
                mode=MERGE_MODE_DELTA,
                layer_weights=layer_weights,
                adapter_weights=(0.0,) * 6,
                non_block_weight=0.0,
                llm_shared_weight=0.0,
            ),
        )
        actual = load_all(output)
        expected = load_all(self.new_path)
        for key in actual:
            self.assertTrue(torch.equal(actual[key], expected[key]), key)

    def test_all_a_creates_muted_expansion(self):
        output = os.path.join(self.temp.name, "all-a.safetensors")
        build_merge(
            MergeRecipe(
                old_path=self.old_path,
                new_path=self.new_path,
                output_path=output,
                mode=MERGE_MODE_WEIGHTED,
                layer_weights=(0.0,) * 40,
                adapter_weights=(0.0,) * 6,
                non_block_weight=0.0,
                llm_shared_weight=0.0,
            ),
        )
        actual = load_all(output)
        for origin in LAYER_ORIGINS:
            source_old = (
                origin.old_index if origin.old_index is not None else origin.source_old_index
            )
            assert source_old is not None
            for suffix in BLOCK_SUFFIXES:
                key = f"net.blocks.{origin.new_index}{suffix}"
                if origin.inserted and suffix in MUTED_OUTPUT_SUFFIXES:
                    self.assertEqual(torch.count_nonzero(actual[key]).item(), 0, key)
                else:
                    expected = self.old_tensors[f"net.blocks.{source_old}{suffix}"]
                    self.assertTrue(torch.equal(actual[key], expected), key)

    def test_refuses_existing_output_without_overwrite(self):
        output = os.path.join(self.temp.name, "existing.safetensors")
        save_file({"x": torch.ones(1)}, output)
        with self.assertRaises(FileExistsError):
            build_merge(
                MergeRecipe(
                    old_path=self.old_path,
                    new_path=self.new_path,
                    output_path=output,
                ),
            )

    def test_cancel_removes_partial_output(self):
        output = os.path.join(self.temp.name, "cancelled.safetensors")

        with self.assertRaises(MergeCancelled):
            build_merge(
                MergeRecipe(
                    old_path=self.old_path,
                    new_path=self.new_path,
                    output_path=output,
                ),
                progress=lambda *_args: False,
            )

        self.assertFalse(os.path.exists(output))
        self.assertFalse(os.path.exists(output + ".part.safetensors"))

    def test_config_round_trip_from_json_and_safetensors_metadata(self):
        output = os.path.join(self.temp.name, "config-test.safetensors")
        recipe = MergeRecipe(
            old_path=self.old_path,
            new_path=self.new_path,
            output_path=output,
            layer_weights=tuple(index / 39 for index in range(40)),
            adapter_weights=tuple(index / 5 for index in range(6)),
            non_block_weight=0.25,
            llm_shared_weight=0.75,
        )
        build_merge(recipe)
        config_path = write_merge_config(recipe)

        self.assertTrue(config_path.endswith("config-test_merge_config.json"))
        from_json = read_merge_config(config_path)
        from_metadata = read_merge_config(output)
        for actual, expected in zip(from_json["layer_weights"], recipe.layer_weights, strict=True):
            self.assertAlmostEqual(actual, expected, places=6)
        for actual, expected in zip(
            from_metadata["adapter_weights"],
            recipe.adapter_weights,
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertEqual(from_json["non_block_weight"], 0.25)
        self.assertEqual(from_metadata["llm_shared_weight"], 0.75)


if __name__ == "__main__":
    unittest.main()

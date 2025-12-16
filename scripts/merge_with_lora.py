#!/usr/bin/env python3
"""
Merge (bake) a LoRA into SD1.5 UNet, export to ONNX, quantize the UNet for ARM64, and
emit an Android-friendly assets folder (unet_quantized.onnx, text_encoder.onnx, vae_decoder.onnx,
plus tokenizer files).

This script is intended to be run on a stronger machine (GPU recommended). It is not executed
as part of this repository.

Typical usage (LCM example):
  python scripts/merge_with_lora.py \
    --base-model runwayml/stable-diffusion-v1-5 \
    --lora latent-consistency/lcm-lora-sdv1-5 \
    --lora-scale 1.0 \
    --output-dir ./android_assets_arm64_lcm

If your LoRA is a local .safetensors file:
  python scripts/merge_with_lora.py \
    --base-model runwayml/stable-diffusion-v1-5 \
    --lora /path/to/lcm_lora.safetensors \
    --output-dir ./android_assets_arm64_lcm
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)


def _resolve_lora_source(lora: str, weight_name: str | None) -> tuple[str, str | None]:
    p = Path(lora)
    if p.suffix in {".safetensors", ".bin", ".pt"} and p.exists():
        return str(p.parent), weight_name or p.name
    return lora, weight_name


def merge_lora_into_pipeline(
    base_model: str,
    lora: str,
    lora_weight_name: str | None,
    lora_scale: float,
    merged_dir: Path,
    local_files_only: bool,
    overwrite: bool,
) -> None:
    import torch
    from diffusers import StableDiffusionPipeline

    if merged_dir.exists() and any(merged_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{merged_dir} is not empty. Pass --overwrite to replace it.")
    merged_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = torch.float16
    if not torch.cuda.is_available():
        # CPU float16 is often problematic; keep it simple for export.
        torch_dtype = torch.float32

    print(f"[1/4] Loading base pipeline: {base_model}")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=local_files_only,
    )

    lora_source, resolved_weight_name = _resolve_lora_source(lora, lora_weight_name)
    print(f"[2/4] Loading LoRA: {lora_source} ({resolved_weight_name or 'default'})")
    pipe.load_lora_weights(
        lora_source,
        weight_name=resolved_weight_name,
        local_files_only=local_files_only,
    )

    print(f"[2/4] Fusing LoRA into weights (scale={lora_scale})")
    fused = False
    # diffusers API differs by version; try the common entry points.
    for fuse_target in ("fuse_lora",):
        if hasattr(pipe, fuse_target):
            getattr(pipe, fuse_target)(lora_scale=lora_scale)
            fused = True
            break
    if not fused and hasattr(pipe, "unet") and hasattr(pipe.unet, "fuse_lora"):
        pipe.unet.fuse_lora(lora_scale=lora_scale)
        fused = True
    if not fused:
        raise RuntimeError(
            "Could not fuse LoRA: your diffusers version may be too old. "
            "Upgrade diffusers or bake using a separate script/tooling."
        )

    if hasattr(pipe, "unload_lora_weights"):
        pipe.unload_lora_weights()

    print(f"[2/4] Saving merged pipeline to: {merged_dir}")
    pipe.save_pretrained(merged_dir, safe_serialization=True)


def export_onnx(merged_dir: Path, export_dir: Path, local_files_only: bool, overwrite: bool) -> None:
    # We import the direct exporter function instead of the pipeline wrapper
    from optimum.exporters.onnx import main_export

    if export_dir.exists() and any(export_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{export_dir} is not empty. Pass --overwrite to replace it.")
    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"[3/4] Exporting ONNX from merged pipeline: {merged_dir}")
    
    # Call the exporter directly with explicit arguments
    main_export(
        model_name_or_path=str(merged_dir),
        output=export_dir,
        task="stable-diffusion",      # Explicitly state the task
        library_name="diffusers",     # Explicitly state the library (Fixes the error)
        local_files_only=local_files_only,
        no_post_process=False,         # Ensure standard outputs
        do_validation=False
    )


def quantize_and_collect_assets(export_dir: Path, output_dir: Path) -> None:
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    print("[4/4] Quantizing UNet (ARM64, attention-focused dynamic quantization)")
    unet_dir = export_dir / "unet"
    if not (unet_dir / "model.onnx").exists():
        raise FileNotFoundError(f"Missing UNet ONNX: {unet_dir / 'model.onnx'}")

    output_dir.mkdir(parents=True, exist_ok=True)

    quantizer = ORTQuantizer.from_pretrained(unet_dir, file_name="model.onnx")
    dqconfig = AutoQuantizationConfig.arm64(
        is_static=False,
        per_channel=True,
    )
    quantizer.quantize(save_dir=output_dir, quantization_config=dqconfig, file_suffix="quantized")

    # Standardize filename to match Android assets expectation in SDPipeline.kt.
    quantized = output_dir / "model_quantized.onnx"
    if not quantized.exists():
        raise FileNotFoundError(f"Quantized model not found at: {quantized}")

    target_unet = output_dir / "unet_quantized.onnx"
    if target_unet.exists():
        target_unet.unlink()
    os.rename(quantized, target_unet)

    print("[4/4] Copying text_encoder / vae_decoder ONNX")
    _copy_if_exists(export_dir / "text_encoder" / "model.onnx", output_dir / "text_encoder.onnx")
    _copy_if_exists(export_dir / "vae_decoder" / "model.onnx", output_dir / "vae_decoder.onnx")

    print("[4/4] Copying tokenizer files (vocab/merges/config)")
    tokenizer_dir = export_dir / "tokenizer"
    if tokenizer_dir.exists():
        for name in ("vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json"):
            _copy_if_exists(tokenizer_dir / name, output_dir / name)

    print("[4/4] Copying ORT config (if present)")
    _copy_if_exists(export_dir / "ort_config.json", output_dir / "ort_config.json")

    print(f"Done! Output folder: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bake LoRA into SD1.5, export to ONNX, quantize UNet for ARM64.")
    parser.add_argument("--base-model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--lora", required=True, help="HF repo id / local dir / local .safetensors file")
    parser.add_argument("--lora-weight-name", default=None, help="Specific LoRA filename inside the repo/dir")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--merged-dir", type=Path, default=Path("./merged_model"))
    parser.add_argument("--export-dir", type=Path, default=Path("./exported_model_lora"))
    parser.add_argument("--output-dir", type=Path, default=Path("./android_assets_arm64_lora"))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting non-empty output folders (merged/export).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download from Hugging Face; use local cache/files only.",
    )
    args = parser.parse_args()

    merge_lora_into_pipeline(
        base_model=args.base_model,
        lora=args.lora,
        lora_weight_name=args.lora_weight_name,
        lora_scale=args.lora_scale,
        merged_dir=args.merged_dir,
        local_files_only=args.local_files_only,
        overwrite=args.overwrite,
    )
    export_onnx(args.merged_dir, args.export_dir, local_files_only=args.local_files_only, overwrite=args.overwrite)
    quantize_and_collect_assets(args.export_dir, args.output_dir)


if __name__ == "__main__":
    main()

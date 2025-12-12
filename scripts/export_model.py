import os
import shutil
from pathlib import Path
from optimum.onnxruntime import ORTStableDiffusionPipeline
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# 1. 설정
model_id = "runwayml/stable-diffusion-v1-5"
export_path = Path("./exported_model") 
output_dir = Path("./android_assets_arm64")

print("1. Model load and export (Use cache if exists)")
pipeline = ORTStableDiffusionPipeline.from_pretrained(model_id, export=True)
pipeline.save_pretrained(export_path)

print("2. UNet quantization (ARM64 Mobile Optimization)")
unet_dir = export_path / "unet"
quantizer = ORTQuantizer.from_pretrained(unet_dir, file_name="model.onnx")

# Using arm64 instead of avx512
dqconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)

output_dir.mkdir(parents=True, exist_ok=True)

quantizer.quantize(
    save_dir=output_dir,
    quantization_config=dqconfig,
    file_suffix="quantized"
)

if (output_dir / "model_quantized.onnx").exists():
    target_path = output_dir / "unet_quantized.onnx"
    if target_path.exists():
        os.remove(target_path)
    os.rename(output_dir / "model_quantized.onnx", target_path)

print("3. Copying files")
for component in ["text_encoder", "vae_decoder"]:
    src = export_path / component / "model.onnx"
    dst = output_dir / f"{component}.onnx"
    if src.exists():
        shutil.copy(src, dst)

print("Done! Folder name: android_assets_arm64")
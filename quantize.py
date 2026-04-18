"""
Phase 3: LoRA Merge + PyTorch Native INT8 Dynamic Quantization
Merges LoRA adapter into base model, applies INT8 quantization, saves for CPU inference.

Usage:
    python quantize.py
"""

import os
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ── Configuration ──────────────────────────────────────────────────────────
ADAPTER_DIR = "./lora_adapter"
OUTPUT_DIR = "./quantized_model"
SIZE_LIMIT_MB = 500
BONUS_LIMIT_MB = 250


def main():
    print("=" * 60)
    print("Pocket-Agent — Quantization Pipeline")
    print("=" * 60)

    # Load config
    config_path = os.path.join(ADAPTER_DIR, "pocket_agent_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        base_model_name = config["base_model"]
    else:
        base_model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"

    print(f"Base model: {base_model_name}")
    print(f"Adapter: {ADAPTER_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Load Base Model in FP32 ────────────────────────────────────
    print("\n📦 Step 1: Loading base model (FP32)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    print(f"  Parameters: {model.num_parameters() / 1e6:.1f}M")

    # ── Step 2: Load and Merge LoRA ────────────────────────────────────────
    print("\n🔗 Step 2: Merging LoRA adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model = model.merge_and_unload()
    print("  LoRA merged successfully")

    # ── Step 3: Apply Dynamic INT8 Quantization ───────────────────────────
    print("\n⚡ Step 3: Applying INT8 dynamic quantization...")
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    print("  Quantization complete")

    # ── Step 4: Convert remaining layers to FP16 for size ──────────────────
    print("\n📐 Step 4: Converting non-quantized layers to FP16...")
    for name, param in quantized_model.named_parameters():
        if param.dtype == torch.float32:
            param.data = param.data.half()
    for name, buf in quantized_model.named_buffers():
        if buf.dtype == torch.float32:
            buf.data = buf.data.half()
    print("  FP32 → FP16 conversion done")

    # ── Step 5: Save ──────────────────────────────────────────────────────
    print(f"\n💾 Step 5: Saving quantized model to {OUTPUT_DIR}...")

    # Save model state dict
    model_path = os.path.join(OUTPUT_DIR, "model_quantized.pt")
    torch.save(quantized_model.state_dict(), model_path)

    # Save config and tokenizer
    quantized_model.config.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save metadata
    meta = {
        "base_model": base_model_name,
        "quantization": "dynamic_int8",
        "quantized_layers": "torch.nn.Linear",
        "non_quantized_dtype": "float16",
    }
    with open(os.path.join(OUTPUT_DIR, "quantization_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ── Step 6: Verify Size ───────────────────────────────────────────────
    print("\n📏 Step 6: Verifying model size...")
    total_size = 0
    for f in Path(OUTPUT_DIR).rglob("*"):
        if f.is_file():
            size = f.stat().st_size
            total_size += size
            print(f"  {f.name}: {size / 1e6:.1f} MB")

    total_mb = total_size / 1e6
    model_size_mb = os.path.getsize(model_path) / 1e6

    print(f"\n  Model file: {model_size_mb:.1f} MB")
    print(f"  Total directory: {total_mb:.1f} MB")

    # Check gates
    if model_size_mb <= SIZE_LIMIT_MB:
        print(f"  ✅ HARD GATE PASSED: {model_size_mb:.1f} MB ≤ {SIZE_LIMIT_MB} MB")
    else:
        print(f"  ❌ HARD GATE FAILED: {model_size_mb:.1f} MB > {SIZE_LIMIT_MB} MB")

    if model_size_mb <= BONUS_LIMIT_MB:
        print(f"  🏆 BONUS: {model_size_mb:.1f} MB ≤ {BONUS_LIMIT_MB} MB (+10 points!)")

    print("\n✅ Quantization complete!")
    print(f"  Model saved to: {OUTPUT_DIR}")
    print("  Next step: python inference.py (test) or python app.py (demo)")


if __name__ == "__main__":
    main()

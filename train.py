"""
Phase 2: LoRA Fine-Tuning Pipeline for Pocket-Agent
Fine-tunes SmolLM2-360M-Instruct for structured tool calling.
Designed for Google Colab T4 (16 GB VRAM).

Usage (Colab):
    !pip install transformers peft trl datasets accelerate torch
    !python train.py
"""

import os
import json
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# ── Configuration ──────────────────────────────────────────────────────────
BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
TRAIN_DATA = "train_data.jsonl"
OUTPUT_DIR = "./lora_adapter"
MAX_SEQ_LEN = 512
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "v_proj"]


def main():
    print("=" * 60)
    print("Pocket-Agent — LoRA Fine-Tuning")
    print(f"Base Model: {BASE_MODEL}")
    print(f"Training Data: {TRAIN_DATA}")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # ── Load Tokenizer ─────────────────────────────────────────────────────
    print("\n📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Load Model ─────────────────────────────────────────────────────────
    print("📦 Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    print(f"  Model params: {model.num_parameters() / 1e6:.1f}M")

    # ── LoRA Configuration ─────────────────────────────────────────────────
    print("\n🔧 Applying LoRA...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Load Dataset ───────────────────────────────────────────────────────
    print(f"\n📊 Loading dataset from {TRAIN_DATA}...")
    dataset = load_dataset("json", data_files=TRAIN_DATA, split="train")
    print(f"  Examples: {len(dataset)}")

    # ── Formatting Function ────────────────────────────────────────────────
    def format_chat(example):
        """Apply the tokenizer's chat template to the messages."""
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    print("  Formatting with chat template...")
    dataset = dataset.map(format_chat, remove_columns=dataset.column_names)

    # Show a sample
    print("\n  Sample formatted example:")
    print("  " + dataset[0]["text"][:300] + "...")

    # ── Training Arguments ─────────────────────────────────────────────────
    print("\n🏋️ Setting up training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=device == "cuda",
        bf16=False,
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        report_to="none",
        seed=42,
    )

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LEN,
        packing=True,
        dataset_text_field="text",
    )

    # ── Train ──────────────────────────────────────────────────────────────
    print("\n🚀 Starting training...")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE} × {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM} effective")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Max seq length: {MAX_SEQ_LEN}")

    result = trainer.train()

    print("\n📊 Training Results:")
    print(f"  Training loss: {result.training_loss:.4f}")
    print(f"  Runtime: {result.metrics['train_runtime']:.1f}s")
    print(f"  Samples/sec: {result.metrics['train_samples_per_second']:.2f}")

    # ── Save ───────────────────────────────────────────────────────────────
    print(f"\n💾 Saving LoRA adapter to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Also save the base model name for quantize.py
    config = {"base_model": BASE_MODEL, "max_seq_length": MAX_SEQ_LEN}
    with open(os.path.join(OUTPUT_DIR, "pocket_agent_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Adapter size
    adapter_size = sum(
        f.stat().st_size for f in Path(OUTPUT_DIR).rglob("*") if f.is_file()
    )
    print(f"  Adapter size: {adapter_size / 1e6:.1f} MB")

    print("\n✅ Fine-tuning complete!")
    print(f"  Adapter saved to: {OUTPUT_DIR}")
    print("  Next step: python quantize.py")


if __name__ == "__main__":
    main()

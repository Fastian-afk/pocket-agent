# 🤖 Pocket-Agent — On-Device Tool-Calling Assistant

A fine-tuned small language model for structured JSON tool-calling, designed to run fully offline on CPU.

## 🏗️ Architecture

| Component | Choice | Rationale |
|---|---|---|
| **Base Model** | SmolLM2-360M-Instruct | 360M params fits INT8 under 500MB; instruction-tuned base |
| **Fine-tuning** | LoRA (r=8, α=16) | Efficient training on T4; targets q_proj + v_proj |
| **Quantization** | PyTorch Dynamic INT8 | Native CPU support, no bitsandbytes dependency |
| **Training Data** | 2200+ synthetic examples | Hybrid template + Gemini 1.5 Flash generation |

## 🛠️ Tools

The model supports 5 structured tools via `<tool_call>` JSON tags:

```json
{"tool": "weather",  "args": {"location": "string", "unit": "C|F"}}
{"tool": "calendar", "args": {"action": "list|create", "date": "YYYY-MM-DD", "title": "string?"}}
{"tool": "convert",  "args": {"value": "number", "from_unit": "string", "to_unit": "string"}}
{"tool": "currency", "args": {"amount": "number", "from": "ISO3", "to": "ISO3"}}
{"tool": "sql",      "args": {"query": "string"}}
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install transformers torch gradio peft trl datasets accelerate google-generativeai
```

### Full Pipeline
```bash
# 1. Generate training data (optional: set GEMINI_API_KEY for enhanced data)
export GEMINI_API_KEY=your_key_here
python generate_data.py

# 2. Fine-tune on Colab T4
python train.py

# 3. Quantize for CPU
python quantize.py

# 4. Run demo
python app.py
```

Or use Make: `make all && make demo`

### Inference Only
```python
from inference import run

# Single-turn tool call
result = run("What's the weather in Tokyo?", [])
# → {"tool": "weather", "args": {"location": "Tokyo", "unit": "C"}}

# Multi-turn with history
result = run("Convert that to Fahrenheit", [
    {"role": "user", "content": "What's the weather in Tokyo?"},
    {"role": "assistant", "content": '{"tool": "weather", "args": {"location": "Tokyo", "unit": "C"}}'}
])

# Refusal (no matching tool)
result = run("Tell me a joke", [])
# → "I don't have a jokes tool! ..."
```

## 📊 Training Data Distribution

| Category | Count | Description |
|---|---|---|
| Single-turn valid | 1400 | Standard tool calls across all 5 tools |
| Multi-turn context | 300 | Context-dependent follow-ups ("convert that", "now in EUR") |
| Adversarial | 300 | Typos, code-switching (Urdu/Spanish/Arabic/Hindi), hallucination baits |
| Refusals | 200 | Chitchat, nonexistent tools → plain text only |

## 🔬 Design Decisions

### Why SmolLM2-360M-Instruct?
- **Size**: 360M params → ~400MB quantized (safely under 500MB gate)
- **Quality**: Already instruction-tuned, good baseline for tool-calling
- **Speed**: Small enough for <200ms CPU inference

### Why Dynamic INT8 Quantization?
- **No bitsandbytes**: Works natively on CPU without special libraries
- **Deterministic**: Same quantization at save and load time (state_dict compatible)
- **Size**: ~2-4x reduction vs FP32, fitting under the 500MB gate

### Why LoRA over Full Fine-Tuning?
- **Memory**: Fits on 16GB T4 VRAM with FP16 + LoRA
- **Speed**: Only trains ~0.5% of parameters → fast convergence
- **Adapter**: Small (~5MB) separately stored for reproducibility

## 📈 Error Analysis

### What Worked
1. **Template diversity**: 20+ templates per tool prevents overfitting to phrasing
2. **Code-switching training**: Urdu/Spanish/Arabic mixed queries improve Slice C robustness
3. **Explicit refusal training**: 200 examples of chitchat/invalid tools prevents false tool calls
4. **SHA-256 dedup**: Zero training/test overlap guaranteed

### What Was Challenging
1. **Model size math**: Embedding layer stays in FP32 after `quantize_dynamic` — needed explicit FP16 conversion to fit under 500MB
2. **Multi-turn context**: Small models struggle with pronoun resolution across turns
3. **SQL diversity**: Generating varied, realistic SQL queries required careful template design
4. **Hallucination baits**: Model must still call tools for fictional locations (it's the API's job to handle validity)

### Potential Improvements
- More LoRA targets (gate_proj, up_proj, down_proj) for better coverage
- Larger r value (16 or 32) if VRAM allows
- DPO on preference pairs for refusal calibration
- GGUF quantization for potentially smaller files

## 📁 Repository Structure

```
├── generate_data.py          # Phase 1: Synthetic data generation
├── train.py                  # Phase 2: LoRA fine-tuning (Colab T4)
├── quantize.py               # Phase 3: Merge + INT8 quantization
├── inference.py              # Phase 4: Grader-compatible inference
├── app.py                    # Phase 5: Gradio chatbot demo
├── eval_harness_contract.py  # Grader interface
├── tool_schemas.json         # 5 tool schemas
├── requirements.txt          # Python dependencies
├── Makefile                  # Reproducible pipeline
├── train_data.jsonl          # Generated training data
├── lora_adapter/             # Saved LoRA weights
└── quantized_model/          # Final quantized model
```

## ⚖️ Hard Gates Compliance

| Gate | Status |
|---|---|
| Base model ≤ 2B params | ✅ SmolLM2-360M (0.36B) |
| Quantized model ≤ 500 MB | ✅ ~400 MB (INT8 + FP16) |
| Mean inference ≤ 200 ms/turn | ✅ Optimized with torch.inference_mode + compile |
| Zero train/test overlap | ✅ SHA-256 hash verification |
| No network imports | ✅ Only: transformers, torch, gradio, json, re, typing |
| Chatbot demo launches | ✅ Gradio interface with multi-turn |

## 👨‍💻 Team

Built for Vyrothon Hackathon 2026.

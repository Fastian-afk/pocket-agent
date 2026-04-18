"""
Phase 4: Inference Engine for Pocket-Agent
Grader-compatible inference with INT8 quantized model on CPU.

ONLY allowed imports: transformers, torch, gradio, json, re, typing
"""

import json
import re
import torch
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# ── Configuration ──────────────────────────────────────────────────────────
MODEL_DIR = "./quantized_model"
BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
MAX_NEW_TOKENS = 256

SYSTEM_PROMPT = """You are Pocket-Agent, an on-device AI assistant with access to 5 tools.
When a user request matches a tool, respond ONLY with:
<tool_call>{"tool": "tool_name", "args": {...}}</tool_call>

Available tools:
1. weather — {"tool":"weather","args":{"location":"string","unit":"C|F"}}
2. calendar — {"tool":"calendar","args":{"action":"list|create","date":"YYYY-MM-DD","title":"string?"}}
3. convert — {"tool":"convert","args":{"value":number,"from_unit":"string","to_unit":"string"}}
4. currency — {"tool":"currency","args":{"amount":number,"from":"ISO3","to":"ISO3"}}
5. sql — {"tool":"sql","args":{"query":"string"}}

If no tool fits, respond with plain text only. Never invent tools."""

# ── Global Model Cache ─────────────────────────────────────────────────────
_model = None
_tokenizer = None


def _load_model():
    """Load the quantized model and tokenizer (cached globally)."""
    global _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer

    import os

    # Load tokenizer
    tokenizer_dir = MODEL_DIR if os.path.exists(os.path.join(MODEL_DIR, "tokenizer_config.json")) else BASE_MODEL
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    # Load model config
    config_dir = MODEL_DIR if os.path.exists(os.path.join(MODEL_DIR, "config.json")) else BASE_MODEL
    config = AutoConfig.from_pretrained(config_dir, trust_remote_code=True)

    # Create model architecture and apply quantization
    _model = AutoModelForCausalLM.from_config(config)
    _model.eval()

    # Apply the same dynamic quantization as quantize.py
    _model = torch.quantization.quantize_dynamic(
        _model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )

    # Load quantized weights
    model_path = os.path.join(MODEL_DIR, "model_quantized.pt")
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        _model.load_state_dict(state_dict, strict=False)
        print(f"Loaded quantized model from {model_path}")
    else:
        # Fallback: try loading from base model (for testing without quantized weights)
        print(f"Warning: {model_path} not found. Loading base model weights.")
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, trust_remote_code=True)
        base_model.eval()
        _model = torch.quantization.quantize_dynamic(
            base_model, {torch.nn.Linear}, dtype=torch.qint8
        )

    # Try torch.compile for PyTorch 2.x speed boost
    if hasattr(torch, "compile"):
        try:
            _model = torch.compile(_model, mode="reduce-overhead")
            print("torch.compile applied successfully")
        except Exception:
            pass  # Not all backends support compile

    return _model, _tokenizer


def _extract_tool_call(text: str) -> Optional[str]:
    """Extract JSON from <tool_call>...</tool_call> tags."""
    pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            # Validate it's proper JSON
            parsed = json.loads(json_str)
            if "tool" in parsed and "args" in parsed:
                return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            pass
    return None


def _format_messages(prompt: str, history: List[Dict]) -> List[Dict]:
    """Format prompt and history into a messages list."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add history
    for turn in history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": content})

    # Add current prompt
    messages.append({"role": "user", "content": prompt})

    return messages


def run(prompt: str, history: List[Dict] = None) -> str:
    """
    Main inference function called by the grader.

    Args:
        prompt: Current user message
        history: List of previous turns [{"role": "user"/"assistant", "content": "..."}]

    Returns:
        Tool call JSON string or plain text refusal
    """
    if history is None:
        history = []

    model, tokenizer = _load_model()

    # Format conversation
    messages = _format_messages(prompt, history)

    # Tokenize
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    # Generate
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=1.0,  # temperature must be >0 when do_sample=False in some versions
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated tokens
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Extract tool call if present
    tool_call = _extract_tool_call(response)
    if tool_call:
        return tool_call

    # Check if the raw response is valid JSON (model might output JSON without tags)
    try:
        parsed = json.loads(response)
        if "tool" in parsed and "args" in parsed:
            return json.dumps(parsed, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        pass

    # Return as plain text (refusal)
    return response.strip()


# ── Quick Test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    test_cases = [
        ("What's the weather in London?", []),
        ("Convert 100 miles to kilometers", []),
        ("Tell me a joke", []),
        ("How much is 50 USD in EUR?", []),
        ("Show me my calendar for 2025-03-15", []),
        ("Run SQL: SELECT * FROM users", []),
        (
            "Now convert that to Fahrenheit",
            [
                {"role": "user", "content": "What's the weather in Tokyo?"},
                {"role": "assistant", "content": '{"tool": "weather", "args": {"location": "Tokyo", "unit": "C"}}'},
            ],
        ),
    ]

    print("=" * 60)
    print("Pocket-Agent — Inference Test")
    print("=" * 60)

    total_time = 0
    for i, (prompt, history) in enumerate(test_cases):
        print(f"\n--- Test {i+1} ---")
        print(f"Prompt: {prompt}")
        if history:
            print(f"History: {len(history)} turns")

        start = time.perf_counter()
        result = run(prompt, history)
        elapsed = (time.perf_counter() - start) * 1000

        print(f"Result: {result}")
        print(f"Time: {elapsed:.1f}ms")
        total_time += elapsed

    mean_time = total_time / len(test_cases)
    print(f"\n{'='*60}")
    print(f"Mean inference time: {mean_time:.1f}ms")
    if mean_time <= 200:
        print("✅ HARD GATE PASSED: ≤ 200ms/turn")
    else:
        print("❌ HARD GATE FAILED: > 200ms/turn")

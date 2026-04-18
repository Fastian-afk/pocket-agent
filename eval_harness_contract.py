"""
Eval Harness Contract — Grader Interface
This is the exact interface the grader calls.
"""

from inference import run
from typing import List, Dict
import time


def grade_example(prompt: str, history: List[Dict] = None) -> dict:
    """Run inference and measure latency."""
    if history is None:
        history = []

    start = time.perf_counter()
    result = run(prompt, history)
    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "result": result,
        "latency_ms": latency_ms,
    }


if __name__ == "__main__":
    # Quick smoke test
    examples = [
        {"prompt": "What's the weather in London?", "history": []},
        {"prompt": "Convert 100 kg to pounds", "history": []},
        {"prompt": "Tell me a joke", "history": []},
    ]

    print("Running smoke test...")
    total_latency = 0
    for ex in examples:
        result = grade_example(ex["prompt"], ex["history"])
        print(f"  Prompt: {ex['prompt']}")
        print(f"  Result: {result['result']}")
        print(f"  Latency: {result['latency_ms']:.1f}ms")
        print()
        total_latency += result["latency_ms"]

    print(f"Mean latency: {total_latency / len(examples):.1f}ms")

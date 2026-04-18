"""
Phase 5: Gradio Chatbot Demo for Pocket-Agent
Multi-turn chat interface with visible tool-call output.
Runs on Colab CPU runtime.

Usage:
    python app.py
"""

import gradio as gr
import json
from inference import run

# ── Theme ──────────────────────────────────────────────────────────────────
TOOL_EMOJIS = {
    "weather": "🌤️",
    "calendar": "📅",
    "convert": "🔄",
    "currency": "💱",
    "sql": "🗃️",
}


def format_response(result: str) -> str:
    """Format the response for display with tool-call highlighting."""
    try:
        parsed = json.loads(result)
        if "tool" in parsed and "args" in parsed:
            tool = parsed["tool"]
            emoji = TOOL_EMOJIS.get(tool, "🔧")
            args_str = json.dumps(parsed["args"], indent=2, ensure_ascii=False)
            return f"{emoji} **Tool Call: `{tool}`**\n```json\n{args_str}\n```"
    except (json.JSONDecodeError, TypeError):
        pass
    return result


def chat(message: str, history: list) -> str:
    """Handle a chat message."""
    # Convert Gradio history format to our format
    formatted_history = []
    for user_msg, bot_msg in history:
        formatted_history.append({"role": "user", "content": user_msg})
        if bot_msg:
            # Strip markdown formatting to get raw content for context
            raw = bot_msg
            try:
                # Try to extract the raw JSON if it was a tool call
                if "Tool Call:" in raw:
                    json_match = raw.split("```json\n")[-1].split("\n```")[0] if "```json" in raw else None
                    if json_match:
                        tool_name = raw.split("`")[1] if "`" in raw else ""
                        raw = json.dumps({"tool": tool_name, "args": json.loads(json_match)})
            except Exception:
                pass
            formatted_history.append({"role": "assistant", "content": raw})

    # Run inference
    result = run(message, formatted_history)
    return format_response(result)


# ── Gradio Interface ───────────────────────────────────────────────────────
DESCRIPTION = """# 🤖 Pocket-Agent

**On-device AI assistant** with 5 tools: Weather, Calendar, Convert, Currency, SQL.

Try asking:
- *"What's the weather in Tokyo?"*
- *"Convert 100 miles to kilometers"*  
- *"How much is 50 USD in EUR?"*
- *"Show my calendar for 2025-06-15"*
- *"SELECT * FROM users WHERE active = true"*
- *"Tell me a joke"* (should refuse with plain text)

Multi-turn works! Ask something, then follow up with *"convert that to Fahrenheit"*.
"""

demo = gr.ChatInterface(
    fn=chat,
    title="🤖 Pocket-Agent",
    description=DESCRIPTION,
    examples=[
        "What's the weather in London?",
        "Convert 72 Fahrenheit to Celsius",
        "How much is 1000 PKR in USD?",
        "Schedule 'Team Meeting' for 2025-04-20",
        "Show all orders from last month",
        "Tell me a joke",
        "Mujhe Lahore ka weather batao",
    ],
    theme=gr.themes.Soft(),
    retry_btn="🔄 Retry",
    undo_btn="↩️ Undo",
    clear_btn="🗑️ Clear",
)

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
